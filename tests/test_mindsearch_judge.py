"""Judge离线测试，只使用临时文件和假模型，不读取真实Key或联网。"""
import json
import sys
from types import SimpleNamespace

import pytest
from judge_mindsearch import evaluate, group_summary, main, parse_verdict, payload, read_cases


def case(index=0, **changes):
    return {'blind_id': index, 'question': 'When?', 'gold': '20 June 1837',
            'prediction': 'June 20, 1837', 'normalized_em': 0,
            'arm': 'mindsearch', 'source_run': 'not shown', 'case_id': str(index), **changes}


def response(verdict='correct', reason='Same date.', finish='stop'):
    return SimpleNamespace(text=json.dumps({'verdict': verdict, 'reason': reason}),
                           finish_reason=finish, n_prompt_pos=100, n_gen_tokens=20)


def test_blind_payload_cache_and_usage_are_not_duplicated(tmp_path):
    calls = []

    def generate(messages):
        data = json.loads(messages[1]['content'])
        assert data == {'question': 'When?', 'gold': '20 June 1837',
                        'prediction': 'June 20, 1837'}
        assert messages[0]['role'] == 'system'
        assert set(data) == {'question', 'gold', 'prediction'}
        calls.append(messages)
        return response()

    out = tmp_path / 'judge'
    summary = evaluate([case(), case(1, arm='single_turn')], generate, out, max_requests=1)
    assert len(calls) == summary['request_count'] == 1
    assert summary['cached_records'] == 1
    assert summary['prompt_tokens_observed'] == 100
    assert summary['completion_tokens_observed'] == 20
    assert summary['human_review_required'] == 2
    rows = [json.loads(x) for x in (out / 'judgments.jsonl').read_text().splitlines()]
    assert all(r['verdict'] == 'correct' and r['em_judge_disagreement'] for r in rows)
    assert rows[0]['request_id'] == rows[1]['request_id']
    assert (out / 'requests.jsonl').stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('text,finish', [
    ('not JSON', 'stop'),
    ('[]', 'stop'),
    ('{"verdict":"correct","reason":""}', 'stop'),
    ('{"verdict":"correct","reason":"x","score":1}', 'stop'),
    ('{"verdict":"maybe","reason":"x"}', 'stop'),
    ('{"verdict":"correct","reason":"x"}', 'length'),
    ('{"verdict":"correct","reason":"x"}', None),
    ('{"verdict":"correct","reason":123}', 'stop'),
])
def test_invalid_or_incomplete_judge_outputs_are_rejected(text, finish):
    with pytest.raises(ValueError):
        parse_verdict(text, finish)


def test_uncertain_error_and_no_gold_are_not_reported_as_incorrect(tmp_path):
    calls = []

    def generate(messages):
        calls.append(messages)
        if len(calls) == 1:
            return response('uncertain', 'Reference is ambiguous.')
        raise TimeoutError('private-provider-message')

    rows = [case(), case(1, prediction='different'), case(2, gold='')]
    out = tmp_path / 'judge'
    summary = evaluate(rows, generate, out)
    stats = summary['by_arm']['mindsearch']
    assert stats['counts'] == {'uncertain': 1, 'error': 1, 'skipped_no_gold': 1}
    assert stats['judge_accuracy_decided'] is None
    assert stats['decision_coverage'] == 0
    assert stats['accuracy_lower_bound'] == 0 and stats['accuracy_upper_bound'] == 1
    assert summary['request_count'] == 2 and summary['usage_may_be_incomplete']
    assert 'private-provider-message' not in (out / 'requests.jsonl').read_text()


def test_parse_failure_keeps_returned_usage_and_is_not_retried(tmp_path):
    def generate(messages):
        r = response()
        r.text = 'bad json'
        return r

    summary = evaluate([case(), case(1)], generate, tmp_path / 'judge')
    assert summary['request_count'] == 1 and summary['cached_records'] == 1
    assert summary['prompt_tokens_observed'] == 100
    assert summary['by_arm']['mindsearch']['counts'] == {'error': 2}


def test_budget_and_existing_output_prevent_requests(tmp_path):
    def forbidden(messages):
        pytest.fail('must not call model')

    with pytest.raises(ValueError):
        evaluate([case(), case(1, prediction='different')], forbidden,
                 tmp_path / 'new', max_requests=1)
    assert not (tmp_path / 'new').exists()
    with pytest.raises(FileExistsError):
        evaluate([case()], forbidden, tmp_path)


def test_source_files_untouched_and_preview_needs_no_keys(tmp_path, monkeypatch, capsys):
    source = tmp_path / 'source'
    source.mkdir()
    original = json.dumps({'case_id': '0', 'question': 'When?', 'gold': '1837',
                           'prediction': '1837', 'arm': 'no_search', 'score': 1}) + '\n'
    (source / 'predictions.jsonl').write_text(original)
    (source / 'run_config.json').write_text('{"git_sha":"source-version","model":"fake"}')
    cases = read_cases([source])
    assert cases[0]['source_git_sha'] == 'source-version'
    assert 'arm' not in payload(cases[0])
    with pytest.raises(ValueError):
        read_cases([source], max_records=0)
    monkeypatch.setattr(sys, 'argv', ['judge_mindsearch.py', str(source)])
    main()
    assert 'Preview only' in capsys.readouterr().out
    assert (source / 'predictions.jsonl').read_text() == original


def test_report_displays_coverage_alongside_accuracy():
    rows = [case(status='ok', verdict='correct'),
            case(status='ok', verdict='incorrect'),
            case(status='ok', verdict='uncertain'),
            case(status='error', verdict=None)]
    report = group_summary(rows)
    assert report['judge_accuracy_decided'] == 0.5 and report['decision_coverage'] == 0.5
    assert report['accuracy_lower_bound'] == 0.25 and report['accuracy_upper_bound'] == 0.75


def test_judge_outputs_are_redacted_before_saving(tmp_path, monkeypatch):
    monkeypatch.setenv('TEST_API_KEY', 'private-credential-example')
    out = tmp_path / 'judge'
    evaluate([case()], lambda messages: response(reason='private-credential-example'), out)
    for path in out.iterdir():
        assert 'private-credential-example' not in path.read_text()


def test_execute_cli_reuses_backend_without_retries_or_redirects(tmp_path, monkeypatch):
    import run_mindsearch
    from lychee_mas.backends import openai_api_backend

    source = tmp_path / 'source'
    source.mkdir()
    (source / 'predictions.jsonl').write_text(json.dumps({
        'case_id': '0', 'question': 'When?', 'gold': '1837', 'prediction': '1837',
        'arm': 'mindsearch', 'score': 1}) + '\n')
    (source / 'run_config.json').write_text('{"git_sha":"source","model":"fake"}')
    out = tmp_path / 'judge'
    clients = []
    calls = []

    class Client:
        def __init__(self, http_client=None):
            self.http_client = http_client
            self.closed = False
            clients.append(self)

        def with_options(self, **options):
            assert options['max_retries'] == 0
            assert options['http_client'].follow_redirects is False
            return Client(options['http_client'])

        def close(self):
            self.closed = True
            if self.http_client:
                self.http_client.close()

    class Backend:
        def __init__(self, **options):
            assert options['model'] == 'fake' and options['temperature'] == 0
            self.client = Client()

        def generate_chat(self, messages, max_new_tokens):
            assert clients[0].closed and not self.client.closed
            assert (out / 'judge_config.json').exists()  # 请求前已保存预算和版本
            assert max_new_tokens == 512
            calls.append(messages)
            return response()

    def configure(args):
        args.model, args.base_url, args.api_key_env = 'fake', 'https://fake.invalid/v1', 'FAKE_KEY'

    monkeypatch.setattr(run_mindsearch, 'configure_api', configure)
    monkeypatch.setattr(openai_api_backend, 'OpenAICompatibleBackend', Backend)
    monkeypatch.setattr(sys, 'argv', ['judge_mindsearch.py', str(source), '--execute',
                                    '--out-dir', str(out), '--max-requests', '1'])
    main()
    assert len(calls) == 1 and all(c.closed for c in clients)
    config = json.loads((out / 'judge_config.json').read_text())
    assert config['same_model_as_any_answer'] is True
    assert config['automatic_retries'] == config['search_calls'] == config['extract_calls'] == 0
