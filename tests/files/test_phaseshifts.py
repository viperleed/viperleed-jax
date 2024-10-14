# Tests for the phaseshifts module.
import pytest

from viperleed.calc.files.phaseshifts import readPHASESHIFTS

from viperleed_jax.files.phaseshifts import Phaseshifts
from viperleed_jax.files.phaseshifts import phaseshift_site_el_order
from viperleed_jax.base_scatterers import SiteEl

@pytest.fixture(scope='session')
def fe2o3_unrelaxed_raw_phaseshifts(fe2o3_unrelaxed_state_after_init,
                                    fe2o3_unrelaxed_input_path):
    slab, rpars = fe2o3_unrelaxed_state_after_init
    _, raw_phaseshifts, _, _ = readPHASESHIFTS(slab, rpars, fe2o3_unrelaxed_input_path/'PHASESHIFTS')
    ps_site_el_map = phaseshift_site_el_order(slab, rpars)
    return raw_phaseshifts, ps_site_el_map

def test_phaseshift_site_el_order(fe2o3_unrelaxed_state_after_init):
    expected_site_el_order = {
        SiteEl('Fe_surf', 'Fe'): 0,
        SiteEl('Fe_def', 'Fe'): 1,
        SiteEl('O_surf', 'O'): 2,
        SiteEl('O_def', 'O'): 3
    }
    assert phaseshift_site_el_order(*fe2o3_unrelaxed_state_after_init) == expected_site_el_order

class TestPhaseshifts:
    """Tests for the Phaseshifts class."""
    def test_build_phaseshifts(self,
                               fe2o3_unrelaxed_raw_phaseshifts,
                               fe2o3_ref_data_fixed_lmax_12):
        raw_phaseshifts, ps_site_el_map = fe2o3_unrelaxed_raw_phaseshifts
        ref_data = fe2o3_ref_data_fixed_lmax_12
        phaseshifts, ps_site_el_map = Phaseshifts(raw_phaseshifts,
                                                  ref_data.energies,
                                                  12,
                                  phaseshift_map=ps_site_el_map)
        assert phaseshifts is not None
