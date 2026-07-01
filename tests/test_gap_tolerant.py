import numpy as np
import pytest
from pyair2stream.config import CommonData
from pyair2stream.model import detect_segments, call_model

def get_base_data(version=4):
    data = CommonData()
    data.version = version
    data.gap_tolerant = True
    data.min_segment_days = 10
    data.mod_num = 'CRN' # Default integrator

    # detect_segments ignores the first 365 days entirely,
    # so we need data longer than 365 to test anything
    n = 365 + 100
    data.n_tot = n
    data.Tair = np.full(n, 20.0)
    data.Q = np.full(n, 10.0)
    data.Twat_obs = np.full(n, 15.0)
    data.Twat_mod = np.full(n, -999.0)
    data.par = np.array([0.01, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]) # simple parameter set
    data.Qmedia = 10.0
    data.flag_par = np.ones(8, dtype=bool)
    data.date = np.zeros((n, 3), dtype=np.int32)
    data.tt = np.zeros(n, dtype=np.float64)
    return data

def test_detect_segments_boundaries():
    """Test min_segment_days boundary behaviors."""
    data = get_base_data()

    # Intentionally drop data
    # We care about indices >= 365.
    data.Tair[365+20:365+25] = -999.0 # Gap of 5 days

    # Gap from 385 to 389 inclusive
    # Valid pre-warmup drop: (365, 384) and (390, 464).
    # Segments:
    # 1: (365, 384) length 20.
    # 2: (390, 464) length 75.
    # Both >= 10, so both should be kept.
    detect_segments(data)

    assert data.segments == [(365, 384), (390, 464)]

    # Test min_segment_days threshold.
    data.Tair[365+10:365+25] = -999.0
    # Gap from 375 to 389 inclusive
    # Valid segments: (365, 374), (390, 464).
    # (365, 374) length 10. So it should be kept!
    detect_segments(data)
    assert data.segments == [(365, 374), (390, 464)]

    # Drop one more day to make it length 9
    data.Tair[365+9:365+25] = -999.0
    # Segment 1 is (365, 373) length 9. It will be dropped.
    detect_segments(data)
    assert data.segments == [(390, 464)]

def test_detect_segments_start_end_gaps():
    """Test gaps at start and end of the record."""
    data = get_base_data()
    data.Tair[365:365+5] = -999.0
    data.Tair[365+90:] = -999.0

    # Valid segment: (370, 454) length 85
    detect_segments(data)
    assert data.segments == [(370, 454)]

def test_detect_segments_v3_v5_ignore_q_gaps():
    """Test that versions 3 and 5 ignore gaps in Discharge (Q)."""
    data_v4 = get_base_data(version=4)
    data_v4.Q[365+10:365+20] = -999.0 # Gap in Q
    detect_segments(data_v4)
    # Segments: (365, 374) length 10 and (385, 464) length 80
    assert data_v4.segments == [(365, 374), (385, 464)]

    data_v3 = get_base_data(version=3)
    data_v3.Q[365+10:365+20] = -999.0 # Same gap in Q
    detect_segments(data_v3)
    # V3 ignores Q, so full series is valid. (365, 464)
    assert data_v3.segments == [(365, 464)]

    data_v5 = get_base_data(version=5)
    data_v5.Q[365+10:365+20] = -999.0
    detect_segments(data_v5)
    assert data_v5.segments == [(365, 464)]

def test_call_model_segmented_state_leakage():
    """Test that state (e.g. simulated Twat) does not leak across gaps."""
    data = get_base_data(version=3) # Simpler, no Q

    # Make Tair varied so Twat changes
    data.Tair[:] = np.linspace(10, 30, data.n_tot)

    # Create a gap: segment 1 ends, segment 2 starts
    data.Tair[365+50:365+60] = -999.0

    detect_segments(data)
    assert data.segments == [(365, 414), (425, 464)]

    call_model(data)

    # Verify the gap contains -999.0
    assert np.all(data.Twat_mod[415:425] == -999.0)
    assert np.all(data.Twat_mod[0:365] == -999.0)

    # Ensure it's valid within the segments
    assert np.all(data.Twat_mod[365:415] != -999.0)
    assert np.all(data.Twat_mod[425:465] != -999.0)

def test_call_model_segmented_reseed_obs():
    """Test re-seeding uses obs if available, else climatology."""
    data = get_base_data()
    data.doy_climatology = np.full(366, 12.0)

    data.Tair[365+40:365+50] = -999.0 # Split into two segments
    detect_segments(data)
    # Segments: (365, 404) and (415, 464)

    # Mock observation at the start of the second segment
    data.Twat_obs[415] = 18.0

    call_model(data)

    assert np.all(data.Twat_mod[415:465] != -999.0)
