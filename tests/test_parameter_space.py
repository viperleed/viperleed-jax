from collections import namedtuple

import pytest
from pytest_cases import case, parametrize_with_cases

from viperleed_jax.parameter_space import ParameterSpace

ParameterSpaceSize = namedtuple(
    'parameter_space_size', ['n_geo', 'n_vib', 'n_occ', 'n_v0r']
)


class CalculatorWithParameterSpaces:
    @case(tags='cu_111')
    def case_default_parameter_space(
        self,
        cu_111_dynamic_l_max_tensor_calculator,
        cu_111_dynamic_l_max_parameter_space,
    ):
        # expected size
        parameter_space_size = ParameterSpaceSize(
            n_v0r=1,
            n_vib=1,
            n_geo=3,
            n_occ=0,
        )
        return (
            cu_111_dynamic_l_max_tensor_calculator,
            cu_111_dynamic_l_max_parameter_space,
            parameter_space_size,
        )

    @case(tags='cu_111')
    def case_more_layers(
        self,
        cu_111_dynamic_l_max_tensor_calculator,
        cu_111_dynamic_l_max_state_after_init,
    ):
        slab, _ = cu_111_dynamic_l_max_state_after_init
        parameter_space = ParameterSpace(slab)
        ## GEOMETRY
        parameter_space.geo_params.fix_layer(4, z_offset=0.0)
        # symmetry constrained xyz movements ± 0.15 A for layer 2
        for param in [
            p
            for p in parameter_space.geo_params.terminal_params
            if p.bound is None
        ]:
            param.set_bound(GeoParamBound(-0.05, +0.05))

        ## VIBRATIONS
        # fix *_def sites (O_def, Fe_def)
        for param in [
            p
            for p in parameter_space.vib_params.terminal_params
            if p.site_element.site.endswith('_def')
        ]:
            parameter_space.vib_params.fix_site_element(
                param.site_element, None
            )  # None fixes to the default value
        # for param in [p for p in parameter_space.vib_params.terminal_params if p.site_element.site.endswith('_surf')]:
        #     parameter_space.vib_params.fix_site_element(param.site_element, None) # None fixes to the default value

        # # the rest can vary ± 0.05 A
        for param in [
            p
            for p in parameter_space.vib_params.terminal_params
            if p.site_element.site.endswith('_surf')
        ]:
            param.set_bound(VibParamBound(-0.05, +0.05))

        ## CHEMISTRY
        # no free parameters
        parameter_space.occ_params.remove_remaining_vacancies()

        # V0R
        # set ± 2 eV
        parameter_space.v0r_param.set_bound(V0rParamBound(-2.0, +2.0))

        # expected size
        parameter_space_size = ParameterSpaceSize(
            n_v0r=1,
            n_vib=1,
            n_geo=12,
            n_occ=0,
        )
        return (
            cu_111_dynamic_l_max_tensor_calculator,
            parameter_space,
            parameter_space_size,
        )

    @pytest.mark.xfail(reason='broken for too small parameter spaces')
    @case(tags='cu_111')
    def case_all_layers(
        self,
        cu_111_dynamic_l_max_tensor_calculator,
        cu_111_dynamic_l_max_state_after_init,
    ):
        slab, _ = cu_111_dynamic_l_max_state_after_init
        parameter_space = ParameterSpace(slab)
        ## GEOMETRY
        # symmetry constrained xyz movements ± 0.15 A for layer 2
        for param in [
            p
            for p in parameter_space.geo_params.terminal_params
            if p.bound is None
        ]:
            param.set_bound(GeoParamBound(-0.05, +0.05))

        ## VIBRATIONS
        # fix *_def sites (O_def, Fe_def)
        for param in [
            p
            for p in parameter_space.vib_params.terminal_params
            if p.site_element.site.endswith('_def')
        ]:
            parameter_space.vib_params.fix_site_element(
                param.site_element, None
            )  # None fixes to the default value
        # for param in [p for p in parameter_space.vib_params.terminal_params if p.site_element.site.endswith('_surf')]:
        #     parameter_space.vib_params.fix_site_element(param.site_element, None) # None fixes to the default value

        # # the rest can vary ± 0.05 A
        for param in [
            p
            for p in parameter_space.vib_params.terminal_params
            if p.site_element.site.endswith('_surf')
        ]:
            param.set_bound(VibParamBound(-0.05, +0.05))

        ## CHEMISTRY
        # no free parameters
        parameter_space.occ_params.remove_remaining_vacancies()

        # V0R
        # set ± 2 eV
        parameter_space.v0r_param.set_bound(V0rParamBound(-2.0, +2.0))
        # expected size
        parameter_space_size = ParameterSpaceSize(
            n_v0r=None,
            n_vib=None,
            n_geo=None,
            n_occ=None,
        )  # TODO: fill
        return cu_111_dynamic_l_max_tensor_calculator, parameter_space

    @pytest.mark.xfail(reason='broken for too small parameter spaces')
    @case(tags='cu_111')
    def case_no_free_vib(
        self,
        cu_111_dynamic_l_max_tensor_calculator,
        cu_111_dynamic_l_max_state_after_init,
    ):
        slab, _ = cu_111_dynamic_l_max_state_after_init
        parameter_space = ParameterSpace(slab)
        ## GEOMETRY
        parameter_space.geo_params.fix_layer(4, z_offset=0.0)
        # symmetry constrained xyz movements ± 0.15 A for layer 2
        for param in [
            p
            for p in parameter_space.geo_params.terminal_params
            if p.bound is None
        ]:
            param.set_bound(GeoParamBound(-0.05, +0.05))

        ## VIBRATIONS
        # fix *_def sites (O_def, Fe_def)
        for param in [
            p
            for p in parameter_space.vib_params.terminal_params
            if p.site_element.site.endswith('_def')
        ]:
            parameter_space.vib_params.fix_site_element(
                param.site_element, None
            )  # None fixes to the default value
        for param in [
            p
            for p in parameter_space.vib_params.terminal_params
            if p.site_element.site.endswith('_surf')
        ]:
            parameter_space.vib_params.fix_site_element(
                param.site_element, None
            )  # None fixes to the default value

        ## CHEMISTRY
        # no free parameters
        parameter_space.occ_params.remove_remaining_vacancies()

        # V0R
        # set ± 2 eV
        parameter_space.v0r_param.set_bound(V0rParamBound(-2.0, +2.0))
        # expected size
        parameter_space_size = ParameterSpaceSize(
            n_v0r=None,
            n_vib=None,
            n_geo=None,
            n_occ=None,
        )  # TODO: fill
        return cu_111_dynamic_l_max_tensor_calculator, parameter_space


@parametrize_with_cases(
    'calculator, parameter_space, expected_size',
    cases=CalculatorWithParameterSpaces,
)
def test_apply_parameter_space(calculator, parameter_space, expected_size):
    # apply the parameter space
    calculator.set_parameter_space(parameter_space)


@parametrize_with_cases(
    'calculator, parameter_space, expected_size',
    cases=CalculatorWithParameterSpaces,
)
def test_expected_parameter_size(calculator, parameter_space, expected_size):
    expected = (
        expected_size.n_v0r,
        expected_size.n_vib,
        expected_size.n_geo,
        expected_size.n_occ,
    )
    assert parameter_space.n_param_split == expected
