"""规范化EM只复用归一化，不继承框架exact的数字/子串捷径。"""
import pytest
from lychee_mas.eval import metrics
from mindsearch_metrics import aggregate_samples, normalized_em, score_answer


@pytest.mark.parametrize('pred,gold,expected', [
    ('James Madison', 'james madison', 1),
    ('  The  Eiffel\nTower! ', 'eiffel tower', 1),
    ('1999', '1998', 0),
    ('21 June 1837', '20 June 1837', 0),
    ('20 June 1837', '20 June 1837', 1),
    ('June 20, 1837', '20 June 1837', 0),
    ('James Madison or Thomas Jefferson', 'James Madison', 0),
    ('Titan IIIE-Centaur', 'Titan IIIE', 0),
    ('', '', 0),
    ('the', 'a', 0),
])
def test_normalized_em(pred, gold, expected):
    assert normalized_em(pred, gold) == expected


def test_legacy_exact_is_preserved_and_other_kinds_are_forwarded():
    assert metrics.score_exact('21 June 1837', '20 June 1837') == 1
    assert score_answer('exact', '21 June 1837', '20 June 1837') == 0
    assert score_answer('aime', '42', '42') == metrics.score('aime', '42', '42')


def test_aggregation_retains_binary_accuracy_and_labels_actual_metric():
    result = aggregate_samples([
        {'score': 1, 'is_correct': True, 'input_positions_total': 10},
        {'score': 0, 'is_correct': False, 'input_positions_total': 20},
    ], {'scorer_kind': 'exact', 'method': 'mindsearch'})
    assert result['accuracy'] == result['mean_score'] == 0.5
    assert result['num_correct'] == 1 and result['total_input_total_positions'] == 30
    assert result['quality_metric'] == result['score_metric'] == 'normalized_em'


def test_offline_rescore_preserves_originals_usage_and_missing_gold(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    path = Path(__file__).parents[1] / 'scripts/rescore_mindsearch.py'
    spec = importlib.util.spec_from_file_location('rescore_test', path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    source = tmp_path / 'source'
    source.mkdir()
    records = [
        {'kind': 'exact', 'prediction': '21 June 1837', 'gold': '20 June 1837',
         'score': 1, 'is_correct': True, 'prompt_tokens': 123},
        {'kind': 'exact', 'prediction': 'unknown', 'gold': '', 'score': None},
    ]
    original = ''.join(json.dumps(r) + '\n' for r in records)
    (source / 'predictions.jsonl').write_text(original)
    (source / 'run_config.json').write_text('{"git_sha":"original-version"}')
    target = tmp_path / 'scored'
    report = script.rescore(source, target)
    result = [json.loads(line) for line in (target / 'predictions.jsonl').read_text().splitlines()]
    assert (source / 'predictions.jsonl').read_text() == original
    assert result[0]['source_score'] == 1 and result[0]['score'] == 0
    assert result[0]['prompt_tokens'] == 123
    assert result[1]['score'] is None and result[1]['is_correct'] is None
    assert report['accuracy'] == 0 and report['n_scored'] == 1
    assert report['source_config']['git_sha'] == 'original-version'
    assert report['real_api_calls'] == 0
    with pytest.raises(FileExistsError):
        script.rescore(source, target)
