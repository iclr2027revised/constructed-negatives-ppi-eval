"""Verify additional frozen summaries and, with --scores, all recovered prediction-level artifacts."""
from pathlib import Path
import argparse, csv, gzip, json, sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from ppi_shift_release import metrics, panels, predictions


def summary_checks():
    base = ROOT / 'results/additional'
    random = pd.read_csv(base / 'FULLY_RANDOM_ALL_96_SEED_ROWS.csv')
    assert len(random) == 96
    assert not random.duplicated(['model', 'seed', 'test_environment']).any()
    for metric in ['auroc', 'auprc']:
        assert np.allclose(random['fully_random_' + metric] - random['uniform_' + metric], random['delta_' + metric], atol=1e-12, rtol=0)
    gold = random[~random.test_environment.str.contains('vidal|intact', case=False)]
    means = gold.groupby(['model', 'test_environment']).delta_auroc.mean()
    assert len(means) == 24 and int((means < 0).sum()) == 21
    nondegree = means[~means.index.get_level_values(1).str.contains('degree')]
    assert len(nondegree) == 20 and int((nondegree < 0).sum()) == 17
    vidal = pd.read_csv(base / 'VIDAL_ALL_STRATA_SEEDS.csv')
    assert len(vidal) == 864
    assert not vidal.duplicated(['cohort', 'model', 'train_environment', 'seed', 'dimension', 'stratum']).any()
    assert (vidal.n_positive + vidal.n_negative == vidal.n_rows).all()
    v = vidal[(vidal.cohort == 'E005') & (vidal.dimension == 'exact_sequence')]
    counts = dict(v.groupby('stratum').n_rows.first())
    assert counts.get('both_novel') == 6035 and counts.get('at_least_one_overlap') == 2107, counts
    identity = dict(vidal[(vidal.cohort == 'E005') & (vidal.dimension == 'protein')].groupby('stratum').n_rows.first())
    assert identity == {'both_novel': 4045, 'at_least_one_overlap': 3786, 'unresolved': 311}, identity
    assert np.allclose(vidal.raw_full_auroc - vidal.full_auroc, vidal.raw_minus_clipped_auroc, atol=1e-12, rtol=0)

    # Verify newly added core summary/provenance tables
    th144 = pd.read_csv(ROOT / 'results/tables/validation_selected_thresholds_144_units.csv', float_precision='round_trip')
    assert len(th144) == 144
    assert set(th144['cohort']) == {'E003_corrected_replay', 'E005_core'}

    shuf512 = pd.read_csv(ROOT / 'results/tables/corrected_512_shuffle_summary.csv')
    assert len(shuf512) == 432

    ds42 = pd.read_csv(ROOT / 'results/tables/dscript_esm2_backbone_provenance_42_units.csv')
    assert len(ds42) == 42
    assert (ds42['inferred_input_embedding_dim'] == 640).all()
    assert (ds42['projection_0_weight_bytes'] == 256000).all()
    assert (ds42['esm2_model_name'] == 'facebook/esm2_t30_150M_UR50D').all()

    return {
        'fully_random_seed_rows': 96,
        'fully_random_gold_wins_uniform': 21,
        'nondegree_wins_uniform': 17,
        'vidal_metric_rows': 864,
        'vidal_sequence_counts': counts,
        'validation_thresholds_units': len(th144),
        'corrected_512_shuffle_summary_cells': len(shuf512),
        'dscript_esm2_150m_provenance_units': len(ds42),
    }


def random_scores():
    entries = {(e.cohort, e.environment): e for e in panels.load_manifest(ROOT) if e.split == 'test'}
    expected = pd.read_csv(ROOT / 'results/additional/FULLY_RANDOM_ALL_96_SEED_ROWS.csv').set_index(['model', 'seed', 'test_environment'])
    with (ROOT / 'manifests/FULLY_RANDOM_PREDICTIONS.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 72
    for row in rows:
        entry = entries[('E005_core', row['test_environment'])]
        path = ROOT / row['path']
        raw = panels.read_rows(path)
        labels = dict(zip(*panels.panel_labels(ROOT, entry)))
        assert all(int(r['interaction_label']) == labels[r['row_id']] for r in raw)
        actual = predictions.evaluate(ROOT, entry, predictions.load_predictions(path), bootstrap_replicates=0)
        target = expected.loc[(row['model'], int(row['seed']), row['test_environment'])]
        for metric in ['auroc', 'auprc']:
            assert abs(actual[metric] - target['fully_random_' + metric]) < 1e-9, (row, metric, actual[metric], target['fully_random_' + metric])
    with (ROOT / 'manifests/FULLY_RANDOM_EXTERNAL_PREDICTIONS.csv').open() as f:
        ext_rows = list(csv.DictReader(f))
    assert len(ext_rows) == 24
    ext_ref_cache = {}
    for row in ext_rows:
        surf = row['test_environment']
        if surf not in ext_ref_cache:
            ref_path = ROOT / f'results/external_pointwise/e005_1024/external__{surf}__all72_models.csv.gz'
            with gzip.open(ref_path, 'rt', newline='') as gf:
                ref_df = pd.read_csv(gf, usecols=['row_id', 'interaction_label'], dtype={'row_id': str})
            ext_ref_cache[surf] = dict(zip(ref_df['row_id'], ref_df['interaction_label'].astype(int)))
        ref_labels = ext_ref_cache[surf]
        with gzip.open(ROOT / row['path'], 'rt', newline='') as gf:
            pred_df = pd.read_csv(gf, dtype={'row_id': str}, float_precision='round_trip')
        assert len(pred_df) == len(ref_labels) and not pred_df['row_id'].duplicated().any()
        assert all(ref_labels[rid] == int(lbl) for rid, lbl in zip(pred_df['row_id'], pred_df['interaction_label']))
        labels = pred_df['interaction_label'].to_numpy(dtype=np.int64)
        scores = pred_df['score'].to_numpy(dtype=np.float64)
        auroc_val = metrics.auroc(labels, scores, score_convention=metrics.CLIPPED_SCORE_CONVENTION)
        auprc_val = metrics.auprc(labels, scores, score_convention=metrics.CLIPPED_SCORE_CONVENTION)
        target = expected.loc[(row['model'], int(row['seed']), surf)]
        assert abs(auroc_val - float(target['fully_random_auroc'])) < 1e-9, (row, 'auroc', auroc_val, target['fully_random_auroc'])
        assert abs(auprc_val - float(target['fully_random_auprc'])) < 1e-9, (row, 'auprc', auprc_val, target['fully_random_auprc'])
    return {
        'fully_random_gold_score_cells': len(rows),
        'fully_random_external_score_cells': len(ext_rows),
        'status': 'PASS',
    }


def recovered_scores():
    # 1. Verify E005_core unshuffled Gold 432 cells against PRIMARY_1024_ALL_GOLD_METRICS.csv and corrected_shuffle_deltas.csv
    entries = {(e.cohort, e.environment): e for e in panels.load_manifest(ROOT) if e.split == 'test'}
    p1024_long = pd.read_csv(ROOT / 'results/additional/PRIMARY_1024_ALL_GOLD_METRICS.csv', float_precision='round_trip')
    p1024 = p1024_long.pivot(
        index=['model_id', 'train_environment', 'seed', 'test_environment'],
        columns='metric',
        values='value',
    )
    shuf_deltas = pd.read_csv(ROOT / 'results/metric_level/corrected_shuffle_deltas.csv', float_precision='round_trip')
    e005_shuf_map = shuf_deltas[shuf_deltas['cohort_label'] == 'E005'].set_index(
        ['model_id', 'train_environment', 'seed', 'test_environment']
    )
    e003_shuf_map = shuf_deltas[shuf_deltas['cohort_label'] == 'E003'].set_index(
        ['model_id', 'train_environment', 'seed', 'test_environment']
    )

    e005_core_verified = 0
    for (m, tenv, s, test_env), target in p1024.iterrows():
        entry = entries[('E005_core', test_env)]
        p_path = ROOT / f'results/pointwise/E005_core/{m}/{tenv}/seed-{int(s)}/{test_env}.csv.gz'
        joined = predictions.join_to_panel(ROOT, entry, predictions.load_predictions(p_path))
        auroc_val = metrics.auroc(joined.labels, joined.scores, score_convention=metrics.CLIPPED_SCORE_CONVENTION)
        auprc_val = metrics.auprc(joined.labels, joined.scores, score_convention=metrics.CLIPPED_SCORE_CONVENTION)
        assert abs(auroc_val - float(target['auroc'])) < 1e-9, (m, tenv, s, test_env, auroc_val, target['auroc'])
        assert abs(auprc_val - float(target['auprc'])) < 1e-9, (m, tenv, s, test_env, auprc_val, target['auprc'])
        s_row = e005_shuf_map.loc[(m, tenv, int(s), test_env)]
        assert abs(auroc_val - float(s_row['auroc_original'])) < 1e-9
        e005_core_verified += 1
    assert e005_core_verified == 432

    # 2. Verify corrected 512-residue shuffled predictions (432 cells: AUROC, AUPRC, MCC, F1, BA)
    shuf512_sum = pd.read_csv(ROOT / 'results/tables/corrected_512_shuffle_summary.csv', float_precision='round_trip').set_index(
        ['model', 'train_environment', 'seed', 'test_environment']
    )
    shuf512_verified = 0
    for test_env in sorted({idx[3] for idx in e003_shuf_map.index}):
        gz_path = ROOT / f'results/shuffle_pointwise/E003_corrected_512_shuffle/test__{test_env}__corrected_512_shuffle_all72_models.csv.gz'
        with gzip.open(gz_path, 'rt', newline='') as gf:
            df = pd.read_csv(gf, float_precision='round_trip')
        assert len(df) == 20222
        labels = df['interaction_label'].to_numpy(dtype=np.int64)
        for (m, tenv, s, t_env), s_row in e003_shuf_map.iterrows():
            if t_env != test_env:
                continue
            col = f'score__{m}__{tenv}__seed-{int(s)}'
            scores = df[col].to_numpy(dtype=np.float64)
            auroc_val = metrics.auroc(labels, scores, score_convention=metrics.CLIPPED_SCORE_CONVENTION)
            auprc_val = metrics.auprc(labels, scores, score_convention=metrics.CLIPPED_SCORE_CONVENTION)
            assert abs(auroc_val - float(s_row['auroc_shuffled'])) < 1e-9, (m, tenv, s, test_env, auroc_val, s_row['auroc_shuffled'])
            sum_row = shuf512_sum.loc[(m, tenv, int(s), test_env)]
            assert abs(auroc_val - float(sum_row['auroc_corrected_shuffle'])) < 1e-9
            assert abs(auprc_val - float(sum_row['auprc_corrected_shuffle'])) < 1e-9
            mcc_val = metrics.mcc_at_threshold(labels, scores, threshold=float(sum_row['threshold']), score_convention=metrics.CLIPPED_SCORE_CONVENTION)
            assert abs(mcc_val - float(sum_row['mcc_corrected_shuffle'])) < 1e-9
            shuf512_verified += 1
    assert shuf512_verified == 432

    # 3. Verify external predictions for E003 (512, 576 cells) and E005 (1024, 576 cells)
    ext_all = pd.read_csv(ROOT / 'results/additional/EXTERNAL_ALL_SEED_METRICS.csv', float_precision='round_trip')
    ext_verified = 0
    for cohort_dir, cohort_name in (('e003_512', 'E003'), ('e005_1024', 'E005')):
        sub_ext = ext_all[ext_all['cohort'] == cohort_name]
        for surf, surf_df in sub_ext.groupby('surface_id'):
            gz_path = ROOT / f'results/external_pointwise/{cohort_dir}/external__{surf}__all72_models.csv.gz'
            with gzip.open(gz_path, 'rt', newline='') as gf:
                df = pd.read_csv(gf, float_precision='round_trip')
            labels = df['interaction_label'].to_numpy(dtype=np.int64)
            for _, erow in surf_df.iterrows():
                col = f"score__{erow['model_id']}__{erow['train_environment']}__seed-{int(erow['seed'])}"
                scores = df[col].to_numpy(dtype=np.float64)
                auroc_val = metrics.auroc(labels, scores, score_convention=metrics.CLIPPED_SCORE_CONVENTION)
                auprc_val = metrics.auprc(labels, scores, score_convention=metrics.CLIPPED_SCORE_CONVENTION)
                assert abs(auroc_val - float(erow['auroc'])) < 1e-9, (cohort_name, surf, col, auroc_val, erow['auroc'])
                assert abs(auprc_val - float(erow['auprc'])) < 1e-9, (cohort_name, surf, col, auprc_val, erow['auprc'])
                ext_verified += 1
    assert ext_verified == 1152

    # 4. Verify paired 512 (E003) and 1024 (E005) predictions on identical frozen 512 test pairs (864 cells)
    matched_df = pd.read_csv(ROOT / 'results/additional/MATCHED_512_1024_ALL_432_SEED_ROWS.csv', float_precision='round_trip')
    common_verified = 0
    for test_env, env_df in matched_df.groupby('test_environment'):
        gz_path = ROOT / f'results/common_512_pointwise/test__{test_env}__paired_e003_e005_all144_models.csv.gz'
        with gzip.open(gz_path, 'rt', newline='') as gf:
            df = pd.read_csv(gf, float_precision='round_trip')
        assert len(df) == 20222
        labels = df['interaction_label'].to_numpy(dtype=np.int64)
        for _, mrow in env_df.iterrows():
            m, tenv, s = mrow['model_id'], mrow['train_environment'], int(mrow['training_seed'])
            c512 = f'e003_512__{m}__{tenv}__seed-{s}'
            c1024 = f'e005_1024__{m}__{tenv}__seed-{s}'
            s512 = df[c512].to_numpy(dtype=np.float64)
            s1024 = df[c1024].to_numpy(dtype=np.float64)
            auroc_512 = metrics.auroc(labels, s512, score_convention=metrics.CLIPPED_SCORE_CONVENTION)
            auprc_512 = metrics.auprc(labels, s512, score_convention=metrics.CLIPPED_SCORE_CONVENTION)
            auroc_1024 = metrics.auroc(labels, s1024, score_convention=metrics.CLIPPED_SCORE_CONVENTION)
            auprc_1024 = metrics.auprc(labels, s1024, score_convention=metrics.CLIPPED_SCORE_CONVENTION)
            assert abs(auroc_512 - float(mrow['e003_auroc'])) < 1e-9
            assert abs(auprc_512 - float(mrow['e003_auprc'])) < 1e-9
            assert abs(auroc_1024 - float(mrow['e005_auroc'])) < 1e-9
            assert abs(auprc_1024 - float(mrow['e005_auprc'])) < 1e-9
            common_verified += 2
    assert common_verified == 864

    # 5. Verify E005 validation predictions (72 units) against validation_selected_thresholds_144_units.csv
    th144 = pd.read_csv(ROOT / 'results/tables/validation_selected_thresholds_144_units.csv', float_precision='round_trip')
    e005_th = th144[th144['cohort'] == 'E005_core']
    val_verified = 0
    for tenv, tdf in e005_th.groupby('train_environment'):
        gz_path = ROOT / f'results/validation_pointwise/E005_core/validation__{tenv}__all12_models.csv.gz'
        with gzip.open(gz_path, 'rt', newline='') as gf:
            df = pd.read_csv(gf, float_precision='round_trip')
        assert len(df) == 47504
        labels = df['interaction_label'].to_numpy(dtype=np.int64)
        for _, urow in tdf.iterrows():
            col = f"score__{urow['model']}__seed-{int(urow['seed'])}"
            scores = df[col].to_numpy(dtype=np.float64)
            t = float(urow['threshold'])
            mcc_val = metrics.mcc_at_threshold(
                labels, scores, threshold=t, score_convention=metrics.CLIPPED_SCORE_CONVENTION
            )
            mcc_ulp = metrics.mcc_at_threshold(
                labels, scores, threshold=float(np.nextafter(t, np.inf)), score_convention=metrics.CLIPPED_SCORE_CONVENTION
            )
            assert min(abs(mcc_val - float(urow['validation_mcc'])), abs(mcc_ulp - float(urow['validation_mcc']))) < 1e-9
            val_verified += 1
    assert val_verified == 72

    return {
        'e005_core_unshuffled_gold_cells_verified': e005_core_verified,
        'e003_corrected_512_shuffle_cells_verified': shuf512_verified,
        'external_prediction_cells_verified': ext_verified,
        'common_512_paired_prediction_cells_verified': common_verified,
        'e005_validation_prediction_units_verified': val_verified,
        'e003_validation_limitation': 'HISTORICAL_E003_PER_ROW_VALIDATION_PREDICTIONS_NEVER_SAVED',
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scores', action='store_true')
    args = p.parse_args()
    result = summary_checks()
    if args.scores:
        result.update(random_scores())
        result.update(recovered_scores())
    print(json.dumps(result, indent=2, default=int))


if __name__ == '__main__':
    main()
