"""离线重算MindSearch预测，输出独立目录，保留旧文件和旧评分。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mindsearch_metrics import metric_name, score_answer


def rescore(source: Path, destination: Path) -> dict:
    records = [json.loads(line) for line in (source / 'predictions.jsonl').read_text().splitlines()
               if line.strip()]
    config = json.loads((source / 'run_config.json').read_text())
    output = []
    for record in records:
        kind = record.get('kind', record.get('scorer_kind', 'exact'))
        gold = record.get('gold')
        score = (None if gold in (None, '') else
                 score_answer(kind, record['prediction'], gold))
        output.append({**record, 'source_score': record.get('score'),
                       'source_score_metric': record.get('score_metric', 'framework_' + kind),
                       'score': score, 'is_correct': None if score is None else score == 1,
                       'score_metric': metric_name(kind)})
    scored = [r for r in output if r['score'] is not None]
    report = {'source_run': str(source.resolve()), 'source_config': config,
              'scoring_method': 'mindsearch_metrics.normalized_em for exact tasks',
              'n': len(output), 'n_scored': len(scored),
              'accuracy': sum(r['score'] for r in scored) / len(scored) if scored else None,
              'real_api_calls': 0}
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'predictions.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in output), encoding='utf-8')
    (destination / 'rescore_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--out-root', type=Path, default=Path('runs/mindsearch_rescored'))
    args = parser.parse_args()
    sources = [p.resolve() for p in args.runs]
    targets = [args.out_root / p.parent.name / p.name for p in sources]
    if len(set(targets)) != len(targets) or any(p.exists() for p in targets):
        parser.error('输出目录已存在或重复；请指定新的--out-root，不覆盖旧结果。')
    for source, target in zip(sources, targets):
        report = rescore(source, target)
        print(f'{source}: normalized scoring accuracy={report["accuracy"]} out_dir={target}')


if __name__ == '__main__':
    main()
