import astropy.units as u
import numpy as np
import pytest
from astropy.timeseries import BinnedTimeSeries, TimeSeries
from specutils import SpectralRegion, Spectrum1D

from padre_meddea import _test_files_directory
from padre_meddea.io.file_tools import (
    read_file,
    read_fits_l0l1_photon,
    read_raw_a0,
)
from padre_meddea.spectrum import spectrum
from padre_meddea.util.pixels import PixelList

f1 = _test_files_directory / "padreMDA0_240916122901.dat"


@pytest.fixture
def ph_list():
    return read_file(f1)


def test_photonlist_slice(ph_list):
    """Test that we can slice a photonlist"""
    start_time = ph_list.time[0]
    stop_time = ph_list.time[2]
    ph_list_slice = ph_list[start_time:stop_time]
    assert isinstance(ph_list_slice, spectrum.PhotonList)
    assert np.all(ph_list_slice.event_list.time >= start_time)
    assert np.all(ph_list_slice.event_list.time <= stop_time)
    # slice with string
    ph_list_slice_str = ph_list[str(start_time) : str(stop_time)]
    assert isinstance(ph_list_slice_str, spectrum.PhotonList)
    assert np.all(ph_list_slice_str.event_list.time >= start_time)
    assert np.all(ph_list_slice_str.event_list.time <= stop_time)


def test_photonlist_slice_invalid(ph_list):
    """Test that we raise an error when slicing a photonlist with invalid types"""
    with pytest.raises(ValueError):
        ph_list[0:1]
    with pytest.raises(ValueError):
        ph_list["2026-07-04T20:31:00", "2026-07-04T20:45:00"]


def test_photonlist_text_summary_contains_event_count(ph_list):
    """Test that PhotonList string summary reports the event count."""
    text_summary = ph_list._text_summary()
    assert (
        "PhotonList (767 events)\n2024-09-16 12:29:01.308 - 12:29:01.774 (0.467s)\n"
        in text_summary
    )


def test_photonlist_add(ph_list):
    """Test that we can add two photonlists together"""
    ph_list2 = ph_list.copy()
    ph_list3 = ph_list + ph_list2
    assert isinstance(ph_list3, spectrum.PhotonList)
    assert len(ph_list3.event_list) == len(ph_list.event_list) + len(
        ph_list2.event_list
    )


@pytest.mark.parametrize(
    "file",
    [_test_files_directory / "padreMDA0_240916122901.dat"]
    + list((_test_files_directory / "eventlist").glob("*.fits")),
)
def test_photonlist(file):
    """Test that we can create a spectrumlist from a raw file"""
    if file.suffix == ".dat":
        phlist = read_raw_a0(file)
    else:
        phlist = read_fits_l0l1_photon(file)

    assert isinstance(phlist, spectrum.PhotonList)

    assert isinstance(phlist.pixel_list, PixelList)
    assert isinstance(phlist.event_list, TimeSeries)
    assert isinstance(phlist.pkt_list, TimeSeries)

    assert len(phlist.event_list) > 0
    assert len(phlist.pkt_list) > 0
    assert phlist.data  # just check that it exists
    assert np.all(phlist.data["event_list"] == phlist.event_list)
    assert np.all(phlist.data["pkt_list"] == phlist.pkt_list)
    assert phlist.calibrated is False
    assert isinstance(phlist.spectrum(pixel_list=phlist.pixel_list), Spectrum1D)
    assert isinstance(phlist.spectrum(), Spectrum1D)

    assert isinstance(
        phlist.lightcurve(
            pixel_list=phlist.pixel_list,
            sr=SpectralRegion([[0, 4000]] * u.pix),
            int_time=0.1 * u.s,
        ),
        BinnedTimeSeries,
    )

    if file.suffix == ".dat":
        assert isinstance(
            phlist.data_rate(),  # need to recreate this fits file to include pktlength
            BinnedTimeSeries,
        )
    # perform basic test on the string representation
    str_list = ["PhotonList", "events", "TimeSeries", "event_list"]
    repr_str = str(phlist)
    for this_str in str_list:
        assert repr_str.count(this_str) >= 1
