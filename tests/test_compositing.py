"""Compositing: metal conservation, run breaks, and hand-worked cases."""
import numpy as np
import pandas as pd
import pytest

from core.compositing import composite_downhole
from core.schema import COLS

HOLE, FROM, TO = COLS['hole'], COLS['from'], COLS['to']


def _samples(rows, hole='H1', col='Au'):
    """rows: (from, to, value) triples, or (hole, from, to, value) quadruples."""
    rows = [r if len(r) == 4 else (hole,) + r for r in rows]
    return pd.DataFrame(rows, columns=[HOLE, FROM, TO, col])


def _regular(n=12, length=1.0, values=None, hole='H1'):
    values = np.arange(n, dtype=float) if values is None else values
    return _samples([(hole, i * length, (i + 1) * length, float(values[i]))
                     for i in range(n)])


# ---- identity ------------------------------------------------------------

@pytest.mark.parametrize('length', [None, 0, -1])
def test_disabled_returns_input_untouched(length):
    s = _regular()
    out = composite_downhole(s, length, value_columns=['Au'])
    pd.testing.assert_frame_equal(out, s)


def test_composite_length_equal_to_support_reproduces_values():
    s = _regular(n=6, length=2.0)
    out = composite_downhole(s, 2.0, value_columns=['Au'])
    assert len(out) == 6
    assert out['Au'].to_numpy() == pytest.approx(s['Au'].to_numpy())
    assert out['interval_m'].to_numpy() == pytest.approx(2.0)


# ---- metal conservation --------------------------------------------------

def test_length_weighted_mean_hand_worked():
    # 1 m at 10, 3 m at 2  ->  (1*10 + 3*2) / 4 = 4
    s = _samples([(0.0, 1.0, 10.0), (1.0, 4.0, 2.0)])
    out = composite_downhole(s, 4.0, value_columns=['Au'])
    assert len(out) == 1
    assert out['Au'].iloc[0] == pytest.approx(4.0)


def test_metal_is_conserved_over_the_whole_hole():
    rng = np.random.default_rng(0)
    s = _regular(n=60, length=1.5, values=rng.lognormal(size=60))
    metal_in = float((s['Au'] * (s[TO] - s[FROM])).sum())
    for length in (3.0, 5.0, 7.0, 12.0):
        out = composite_downhole(s, length, value_columns=['Au'])
        metal_out = float((out['Au'] * out['interval_m']).sum())
        assert metal_out == pytest.approx(metal_in), f'at {length} m'


def test_interval_straddling_a_boundary_is_split_proportionally():
    # One 4 m interval across a 2 m grid contributes 2 m to each composite.
    s = _samples([(0.0, 4.0, 8.0)])
    out = composite_downhole(s, 2.0, value_columns=['Au'])
    assert len(out) == 2
    assert out['interval_m'].to_numpy() == pytest.approx([2.0, 2.0])
    assert out['Au'].to_numpy() == pytest.approx([8.0, 8.0])


def test_composite_averages_only_within_its_own_cell():
    s = _samples([(0.0, 2.0, 0.0), (2.0, 4.0, 10.0)])
    out = composite_downhole(s, 2.0, value_columns=['Au'])
    assert out['Au'].to_numpy() == pytest.approx([0.0, 10.0])


# ---- variance reduction, the reason this module exists -------------------

def test_compositing_reduces_noise_on_a_flat_field():
    rng = np.random.default_rng(1)
    n = 4000
    s = _regular(n=n, length=1.0, values=rng.normal(0.0, 1.0, n))
    out = composite_downhole(s, 10.0, value_columns=['Au'])
    # Ten independent 1 m samples per composite => sd shrinks by ~sqrt(10).
    assert out['Au'].std() == pytest.approx(1.0 / np.sqrt(10.0), rel=0.15)


# ---- runs, gaps and hole boundaries --------------------------------------

def test_holes_never_composite_together():
    s = _samples([('H1', 0.0, 2.0, 1.0), ('H2', 0.0, 2.0, 5.0)])
    out = composite_downhole(s, 10.0, value_columns=['Au'])
    assert len(out) == 2
    assert set(out[HOLE]) == {'H1', 'H2'}


def test_wide_gap_splits_the_run():
    s = _samples([(0.0, 2.0, 1.0), (100.0, 102.0, 5.0)])
    out = composite_downhole(s, 10.0, value_columns=['Au'])
    assert len(out) == 2
    assert out[FROM].to_numpy() == pytest.approx([0.0, 100.0])


def test_narrow_gap_stays_inside_a_composite_and_lowers_coverage():
    # 1 m assayed, 1 m gap, 1 m assayed, inside a 10 m composite.
    s = _samples([(0.0, 1.0, 4.0), (2.0, 3.0, 6.0)])
    out = composite_downhole(s, 10.0, value_columns=['Au'])
    assert len(out) == 1
    row = out.iloc[0]
    assert row['interval_m'] == pytest.approx(2.0)     # assayed length
    assert (row[TO] - row[FROM]) == pytest.approx(3.0)  # extent spans the gap
    assert row['coverage'] == pytest.approx(0.2)
    assert row['Au'] == pytest.approx(5.0)


def test_each_run_anchors_its_own_grid():
    s = _samples([(7.0, 9.0, 1.0), (500.0, 502.0, 2.0)])
    out = composite_downhole(s, 10.0, value_columns=['Au'])
    assert out[FROM].to_numpy() == pytest.approx([7.0, 500.0])


def test_min_coverage_drops_sparse_composites():
    s = _samples([(0.0, 10.0, 1.0), (200.0, 201.0, 9.0)])
    out = composite_downhole(s, 10.0, value_columns=['Au'], min_coverage=0.5)
    assert len(out) == 1
    assert out['Au'].iloc[0] == pytest.approx(1.0)


# ---- tail handling -------------------------------------------------------

def test_short_tail_merges_into_the_previous_composite():
    # 11 m of ground on a 10 m grid: the 1 m tail joins the first composite.
    s = _regular(n=11, length=1.0, values=np.zeros(11))
    out = composite_downhole(s, 10.0, value_columns=['Au'])
    assert len(out) == 1
    assert out['interval_m'].iloc[0] == pytest.approx(11.0)


def test_long_tail_stands_alone():
    s = _regular(n=18, length=1.0, values=np.zeros(18))
    out = composite_downhole(s, 10.0, value_columns=['Au'])
    assert len(out) == 2
    assert out['interval_m'].to_numpy() == pytest.approx([10.0, 8.0])


def test_run_shorter_than_one_composite_is_kept_whole():
    s = _samples([(0.0, 2.0, 3.0)])
    out = composite_downhole(s, 50.0, value_columns=['Au'])
    assert len(out) == 1
    assert out['interval_m'].iloc[0] == pytest.approx(2.0)
    assert out['Au'].iloc[0] == pytest.approx(3.0)


# ---- missing assays ------------------------------------------------------

def test_missing_values_are_excluded_from_the_weighting():
    s = _samples([(0.0, 1.0, 10.0), (1.0, 2.0, np.nan), (2.0, 3.0, 20.0)])
    out = composite_downhole(s, 10.0, value_columns=['Au'])
    assert out['Au'].iloc[0] == pytest.approx(15.0)   # not 10.0, not 30/3
    assert out['interval_m'].iloc[0] == pytest.approx(3.0)


def test_composite_with_no_assayed_value_is_nan_not_zero():
    s = _samples([(0.0, 2.0, np.nan)])
    out = composite_downhole(s, 10.0, value_columns=['Au'])
    assert np.isnan(out['Au'].iloc[0])


def test_columns_are_weighted_independently():
    s = pd.DataFrame({
        HOLE: ['H1', 'H1'], FROM: [0.0, 1.0], TO: [1.0, 3.0],
        'Ag': [10.0, 40.0], 'Pb': [np.nan, 2.0]})
    out = composite_downhole(s, 10.0, value_columns=['Ag', 'Pb'])
    assert out['Ag'].iloc[0] == pytest.approx(30.0)   # (1*10 + 2*40) / 3
    assert out['Pb'].iloc[0] == pytest.approx(2.0)    # 2 m of Pb only


# ---- geometry ------------------------------------------------------------

def test_midpoint_is_the_assayed_centroid_not_the_cell_centre():
    # All the material sits in the first 2 m of a 10 m cell.
    s = _samples([(0.0, 2.0, 1.0)])
    out = composite_downhole(s, 10.0, value_columns=['Au'])
    assert out['mid_m'].iloc[0] == pytest.approx(1.0)


def test_output_is_ordered_and_carries_the_pipeline_columns():
    s = _samples([('H2', 0.0, 2.0, 1.0), ('H1', 40.0, 42.0, 1.0),
                  ('H1', 0.0, 2.0, 1.0)])
    out = composite_downhole(s, 5.0, value_columns=['Au'])
    assert list(out[HOLE]) == ['H1', 'H1', 'H2']
    assert out[FROM].to_numpy() == pytest.approx([0.0, 40.0, 0.0])
    for col in (HOLE, FROM, TO, 'interval_m', 'mid_m', 'Au'):
        assert col in out.columns
    assert (out['n_samples'] == 1).all()


def test_unsorted_input_gives_the_same_answer_as_sorted():
    rng = np.random.default_rng(2)
    s = _regular(n=40, length=1.5, values=rng.lognormal(size=40))
    a = composite_downhole(s, 6.0, value_columns=['Au'])
    b = composite_downhole(s.sample(frac=1.0, random_state=3), 6.0,
                           value_columns=['Au'])
    pd.testing.assert_frame_equal(a, b)


def test_unassayed_ground_inherits_grade_and_never_counts_as_zero():
    # 1 m at 10, 1 m unassayed. The composite is 10 over 2 m of assayed
    # length -- not 5, which is what treating the null as zero would give.
    s = _samples([(0.0, 1.0, 10.0), (1.0, 2.0, np.nan)])
    out = composite_downhole(s, 10.0, value_columns=['Au'])
    assert out['Au'].iloc[0] == pytest.approx(10.0)
    assert out['interval_m'].iloc[0] == pytest.approx(2.0)


def test_metal_conserved_exactly_when_every_interval_is_assayed():
    rng = np.random.default_rng(4)
    s = _regular(n=200, length=1.3, values=rng.lognormal(size=200))
    metal_in = float((s['Au'] * (s[TO] - s[FROM])).sum())
    for length in (2.0, 3.0, 5.0, 8.0, 12.0, 20.0):
        out = composite_downhole(s, length, value_columns=['Au'])
        assert float((out['Au'] * out['interval_m']).sum()) == \
            pytest.approx(metal_in, rel=1e-12), f'at {length} m'
