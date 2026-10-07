"""Independent arithmetic/lineage audit of completed simulation outputs.

No production search/prediction helpers or thermal simulations are imported.
This is not an independent physical model or experimental validation.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from audit_completed_stages import audit as audit_early, canonical, sha

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE / 'upstream_package'
DATA = ROOT / 'results_upgrade_20261006'
CP = ROOT / '.upgrade_checkpoints_20261006'
OUT = PACKAGE / 'analysis/verified_complete_results'


def audit():
    early = audit_early()
    OUT.mkdir(exist_ok=True)
    checks = []
    def check(name, passed, **context):
        checks.append(dict(check=name, passed=bool(passed), **context))
    protocol = json.loads((ROOT/'config/upgrade_design_20261006.json').read_text())
    config = json.loads((ROOT/'config/study_v18.json').read_text())
    signed = json.loads((DATA/'SIGNED_RECOVERY_IDENTITY.json').read_text())
    original = json.loads((DATA/'INPUT_IDENTITY.json').read_text())
    amendment = json.loads((ROOT/'config/upgrade_energy_confirmation_amendment_20261006.json').read_text())
    addon = json.loads((DATA/'07_AMENDMENT_INPUT_IDENTITY.json').read_text())
    for name, expected in signed['files'].items():
        check('signed_locked_input_unchanged', sha(ROOT/name) == expected, source=name)
    expected_addon = canonical(dict(original=original['identity'],
        amendment=sha(ROOT/'config/upgrade_energy_confirmation_amendment_20261006.json'),
        script=sha(ROOT/'scripts/run_upgrade_energy_confirmation_20261006.py'),
        refined_count_source=sha(ROOT/'results_v18/decision_grid_refinement_v16.csv')))
    check('amendment_identity_unchanged', expected_addon == addon['identity'])
    stage_rows = {}
    for stage, identity, csv_name, total in [
        ('05_signed_configuration_model_stress',signed['identity'],'05_configuration_model_stress.csv',120),
        ('07_energy_controlled_fine_grid',addon['identity'],'07_energy_controlled_fine_grid.csv',48),
        ('08_refined_configuration_inverse',addon['identity'],'08_refined_configuration_inverse.csv',104)]:
        receipt = json.loads((DATA/(stage+'_receipt.json')).read_text())
        check('receipt_identity', receipt.get('identity',receipt.get('input_identity')) == identity, stage=stage)
        check('receipt_csv_checksum', receipt['csv_sha256'] == sha(DATA/csv_name), stage=stage)
        paths = sorted((CP/stage).glob('*.json'))
        check('complete_checkpoint_coverage', len(paths) == total == receipt['tasks'], stage=stage)
        records = []
        for path in paths:
            obj = json.loads(path.read_text())
            check('checkpoint_identity_and_digest', obj.get('identity',obj.get('input_identity')) == identity and obj['result_hash'] == canonical(obj['result']), stage=stage, source=path.name)
            if 'task_hash' in obj:
                check('checkpoint_task_filename', obj['task_hash'] == path.stem, stage=stage)
            records.append((path.stem,obj['result']))
        stage_rows[stage] = records
        frame = pd.read_csv(DATA/csv_name)
        keys = ['design_id','power_MW','energy_to_power_h','scenario_id'] if stage.startswith('05') else (['design_id','power_MW','energy_to_power_h','n_cells'] if stage.startswith('07') else ['design_id','service_id','n_cells'])
        check('unique_case_grain', len(frame)==total and not frame.duplicated(keys).any(), stage=stage)

    stress = []
    raw_audits = []
    migrated = 0
    minimum_fraction = config['admissibility']['minimum_high_grade_discharge_fraction']
    for key, row in stage_rows['05_signed_configuration_model_stress']:
        trajectory = row['trajectory']
        counts = [x['count'] for x in trajectory]
        check('stress_unskipped_enumeration', counts == list(range(row['power_lower_count'],row['power_lower_count']+len(counts))) and counts[-1] <= row['maximum_count'], case=key)
        check('stress_energy_requirement', math.isclose(row['required_energy_MWh'],row['power_MW']*row['energy_to_power_h'],rel_tol=1e-12),case=key)
        feasible = []
        for item in trajectory:
            energy = item['installed_energy_MWh']
            flag = item['candidate_admissible'] and energy+1e-12 >= row['required_energy_MWh']
            feasible.append(flag)
            check('stress_energy_margin', math.isclose(item['energy_margin_MWh'],energy-row['required_energy_MWh'],abs_tol=1e-10),case=key)
            check('stress_feasibility', item['feasible']==flag,case=key)
            if 'migration' in row:
                continue
            domain = item['temperature_min_K'] >= item['property_min_K']-1e-3 and item['temperature_max_K'] <= item['property_max_K']+1e-3
            physical = item['pressure_drop_kPa'] <= protocol['maximum_bed_pressure_drop_kPa'] and item['high_grade_discharge_fraction'] >= minimum_fraction
            check('whole_cycle_property_domain', domain == item['property_domain_supported'],case=key)
            check('stress_admissibility', item['candidate_admissible']==(physical and domain),case=key)
            check('installed_energy_from_module', math.isclose(energy,item['count']*item['deliverable_energy_MWh'],rel_tol=1e-12,abs_tol=1e-12),case=key)
            initial = item['charge_reference_inventory_J']
            residual = item['discharge_boundary_energy_J']+item['discharge_final_reference_inventory_J']+item['discharge_wall_loss_J']-initial
            denominator = initial if initial > 0 else max(abs(initial),item['reference_span_energy_J']*1e-12,1e-12)
            error = 100*residual/denominator
            check('independent_signed_discharge_residual', math.isclose(error,item['discharge_energy_balance_error_pct'],rel_tol=1e-7,abs_tol=2e-10),case=key)
            check('unchanged_energy_balance_tolerance', max(abs(error),abs(item['charge_energy_balance_error_pct'])) <= .1,case=key)
            integer_path = CP/'signed_integer_evaluations'/key/f"{item['count']:06d}.json"
            record = json.loads(integer_path.read_text())
            check('persisted_integer_identity_and_digest', record['identity']==signed['identity'] and record['result_hash']==canonical(record['result']),case=key)
            check('persisted_integer_matches_stage_trajectory', all(item[k]==v for k,v in record['result'].items()),case=key)
            raw_audits.append(dict(case=key,count=item['count'],scenario_id=row['scenario_id'],fluid_model_id=row['fluid_model_id'],initial_reference_inventory_J=initial,independent_discharge_error_pct=error,property_domain_supported=domain))
        check('no_feasible_prefix_skipped', not any(feasible[:-1]),case=key)
        expected = 'FEASIBLE' if feasible[-1] else ('MODEL_DOMAIN_UNSUPPORTED' if trajectory[-1].get('property_domain_supported') is False else 'NO_FEASIBLE_COUNT_IN_REGISTERED_DOMAIN')
        check('stress_outcome_classification', row['status']==expected,case=key)
        if row['status']=='FEASIBLE':
            check('stress_first_feasible_count', row['minimum_count']==counts[-1] and row['exhaustive_minimality_verified'],case=key)
        else:
            check('unresolved_not_promoted', row['minimum_count'] is None and not row['exhaustive_minimality_verified'],case=key)
            if row['status']=='NO_FEASIBLE_COUNT_IN_REGISTERED_DOMAIN':
                check('infeasible_full_registered_domain', counts[-1]==row['maximum_count'],case=key)
        if 'migration' in row:
            candidates = [p for p in (CP/'05_configuration_model_stress').glob('*.json') if sha(p)==row['original_checkpoint_sha256']]
            check('migration_original_preserved', len(candidates)==1,case=key)
            old = json.loads(candidates[0].read_text())
            check('migration_original_integrity', old['input_identity']==original['identity'] and old['result_hash']==canonical(old['result']),case=key)
            check('migration_values_unchanged', all(row[k]==v for k,v in old['result'].items()),case=key)
            migration_scenario = next(s for s in protocol['robustness_scenarios'] if s['scenario_id']==row['scenario_id'])
            check('migration_no_loss_only', migration_scenario['surface_coefficient_W_m2K']==0 and migration_scenario['axial_dispersion_m2_s']==0 and migration_scenario['solid_axial_diffusivity_m2_s']==0,case=key)
            migrated += 1
        decreases = sum(b['installed_energy_MWh'] < a['installed_energy_MWh']-1e-10 for a,b in zip(trajectory,trajectory[1:]))
        check('nonmonotone_increments_independent', decreases==row['nonmonotone_increments'],case=key)
        stress.append({k:v for k,v in row.items() if not isinstance(v,(list,dict))})
    check('migration_count', migrated==24)
    stress = pd.DataFrame(stress)
    raw_audits = pd.DataFrame(raw_audits)
    raw_audits.to_csv(OUT/'independent_signed_inventory_audit.csv',index=False)
    stress.groupby(['scenario_id','fluid_model_id','status']).size().rename('cases').reset_index().to_csv(OUT/'stress_outcomes_all_cases.csv',index=False)

    committed = json.loads((DATA/'07_PREDICTION_COMMITMENT.json').read_text())
    check('prediction_commitment_csv_unchanged', sha(DATA/'07_prospective_predictions_before_confirmation.csv')==committed['csv_sha256'])
    commit_time = datetime.fromisoformat(committed['committed']).timestamp()
    first_reference = min(p.stat().st_mtime for p in (CP/'07_energy_controlled_fine_grid').glob('*.json'))
    check('predictions_saved_before_references', commit_time < first_reference,seconds=first_reference-commit_time)
    curves = pd.read_csv(DATA/'02_prior_duty_curves.csv')
    predictions = pd.read_csv(DATA/'07_prospective_predictions_before_confirmation.csv')
    predicted_map = {}
    independent_predictions = []
    for row in predictions.to_dict('records'):
        curve = curves[curves.design_id==row['design_id']].sort_values('unit_power_MW')
        nominal = curve.nominal_thermal_input_kW.iloc[0]/1000
        lower = max(1,math.ceil(row['power_MW']/nominal))
        upper = math.floor(row['power_MW']/(nominal*.05))
        x = curve.unit_power_MW.to_numpy()
        interps = [PchipInterpolator(x,curve[c],extrapolate=False) for c in ['deliverable_energy_MWh','pressure_drop_kPa','high_grade_discharge_fraction']]
        candidate = linear_candidate = None
        predicted_energy = None
        for count in range(lower,upper+1):
            q = row['power_MW']/count
            check('supplemental_no_duty_extrapolation', x[0]-1e-12 <= q <= x[-1]+1e-12)
            e,p,f = [float(fun(q)) for fun in interps]
            le,lp,lf = [float(np.interp(q,x,curve[c])) for c in ['deliverable_energy_MWh','pressure_drop_kPa','high_grade_discharge_fraction']]
            if candidate is None and count*e+1e-12 >= row['power_MW']*row['energy_to_power_h'] and p<=50 and f>=minimum_fraction:
                candidate,predicted_energy = count,count*e
            if linear_candidate is None and count*le+1e-12 >= row['power_MW']*row['energy_to_power_h'] and lp<=50 and lf>=minimum_fraction:
                linear_candidate = count
        check('supplemental_prediction_independent', candidate==row['predicted_count'] and math.isclose(predicted_energy,row['predicted_installed_energy_MWh'],rel_tol=1e-12),case=row['design_id'])
        predicted_map[(row['design_id'],row['power_MW'],row['energy_to_power_h'])] = candidate
        independent_predictions.append(dict(row,linear_count=linear_candidate))
    predicted_digest_rows = [{k:r[k] for k in ['design_id','power_MW','energy_to_power_h','predicted_count','predicted_installed_energy_MWh','prediction_status']} for r in predictions.to_dict('records')]
    # CSV round trips may lose an ulp; exact semantic digest belongs to the
    # original driver commitment. Independently verify persisted CSV and numbers.
    confirmation_rows = []
    for key,row in stage_rows['07_energy_controlled_fine_grid']:
        trajectory = row['trajectory']
        counts = [x['count'] for x in trajectory]
        check('supplemental_unskipped_integer_prefix', counts==list(range(row['power_lower_count'],row['minimum_count']+1)),case=key)
        flags = [x['candidate_admissible'] and x['installed_energy_MWh']+1e-12>=row['required_energy_MWh'] for x in trajectory]
        check('supplemental_first_feasible_integer', flags==[x['feasible'] for x in trajectory] and not any(flags[:-1]) and flags[-1],case=key)
        check('supplemental_energy_and_grade', all(math.isclose(x['energy_margin_MWh'],x['installed_energy_MWh']-row['required_energy_MWh'],abs_tol=1e-10) for x in trajectory) and row['selected_fraction']>=minimum_fraction and row['selected_pressure_drop_kPa']<=50,case=key)
        prediction = predicted_map[(row['design_id'],row['power_MW'],row['energy_to_power_h'])]
        confirmation_rows.append(dict(design_id=row['design_id'],power_MW=row['power_MW'],energy_to_power_h=row['energy_to_power_h'],n_cells=row['n_cells'],predicted_count=prediction,observed_count=row['minimum_count'],power_lower_count=row['power_lower_count'],count_error=prediction-row['minimum_count'],installed_energy_MWh=row['installed_energy_MWh']))
    confirmation = pd.DataFrame(confirmation_rows)
    confirmation.to_csv(OUT/'independent_high_energy_confirmation.csv',index=False)
    pd.DataFrame(independent_predictions).to_csv(OUT/'independent_high_energy_predictions.csv',index=False)
    paired = confirmation[confirmation.n_cells==2048].merge(confirmation[confirmation.n_cells==4096],on=['design_id','power_MW','energy_to_power_h'],suffixes=('_2048','_4096'),validate='one_to_one')

    reference = pd.read_csv(DATA/'08_refined_configuration_reference.csv')
    reference_map = {(r['design_id'],r['scenario_id']):r for r in reference.to_dict('records')}
    inverse_rows, replacement_rows = [],[]
    for key,row in stage_rows['08_refined_configuration_inverse']:
        ref = reference_map[(row['design_id'],row['service_id'])]
        check('refined_target_count_and_service', row['baseline_count']==ref['parallel_unit_count'] and row['candidate_count']==ref['parallel_unit_count']-1 and row['required_energy_MWh']==ref['required_energy_MWh'],case=key)
        expected_gain = 100*max(0,row['required_energy_MWh']/row['candidate_installed_energy_MWh']-1)
        check('refined_inverse_target_arithmetic', math.isclose(expected_gain,row['required_qualified_energy_gain_pct'],abs_tol=1e-11) and math.isclose(row['candidate_energy_margin_MWh'],row['candidate_installed_energy_MWh']-row['required_energy_MWh'],abs_tol=1e-10),case=key)
        check('complete_five_material_replacements', len(row['replacement_tests'])==5 and {x['fluid_model_id'] for x in row['replacement_tests']}==set(config['fluid_model_ids']),case=key)
        for item in row['replacement_tests']:
            e = item['installed_energy_MWh']+1e-12>=row['required_energy_MWh']
            a = item['pressure_drop_kPa']<=50 and item['high_grade_fraction']>=minimum_fraction
            check('refined_replacement_admissibility_and_energy', e==item['energy_feasible'] and a==item['candidate_admissible'] and item['reduced_count_feasible']==(e and a and not row['unchanged_rating_power_blocked']),case=key)
            check('refined_replacement_relative_energy', math.isclose(item['relative_energy_change_pct'],100*(item['installed_energy_MWh']/row['candidate_installed_energy_MWh']-1),abs_tol=1e-10),case=key)
            replacement_rows.append(dict(case_design_id=row['design_id'],service_id=row['service_id'],n_cells=row['n_cells'],baseline_fluid=row['fluid_model_id'],**item))
        inverse_rows.append({k:v for k,v in row.items() if not isinstance(v,(list,dict))})
    inverse = pd.DataFrame(inverse_rows)
    replacements = pd.DataFrame(replacement_rows)
    compare_keys = ['case_design_id','service_id','fluid_model_id']
    replacement_pairs = replacements[replacements.n_cells==2048].merge(replacements[replacements.n_cells==4096],on=compare_keys,suffixes=('_2048','_4096'),validate='one_to_one')
    replacement_pairs['grid_energy_change_pct'] = 100*(replacement_pairs.installed_energy_MWh_4096/replacement_pairs.installed_energy_MWh_2048-1)
    replacement_pairs['feasibility_changed'] = replacement_pairs.reduced_count_feasible_2048 != replacement_pairs.reduced_count_feasible_4096
    replacement_pairs['same_material'] = replacement_pairs.fluid_model_id == replacement_pairs.baseline_fluid_2048
    replacement_pairs.to_csv(OUT/'independent_refined_material_grid_comparison.csv',index=False)
    inverse.to_csv(OUT/'independent_refined_inverse_targets.csv',index=False)
    summary = dict(status='COMPLETED_COMPUTATION_ARITHMETIC_REVIEW_NOT_PUBLICATION_ACCEPTANCE',checked_at_utc=datetime.now(timezone.utc).isoformat(),checks=len(checks),failed_checks=sum(not c['passed'] for c in checks),early_stage_checks=early['checks'],stress_cases=len(stress),stress_outcomes=stress.status.value_counts().to_dict(),signed_raw_integer_evaluations=len(raw_audits),negative_reference_inventory_evaluations=int((raw_audits.initial_reference_inventory_J<=0).sum()),maximum_signed_discharge_error_pct=float(raw_audits.independent_discharge_error_pct.abs().max()),stress_cases_with_nonmonotone_installed_energy=int((stress.nonmonotone_increments>0).sum()),supplemental_cases=len(predictions),supplemental_predictions_matching_2048=int((confirmation[confirmation.n_cells==2048].count_error==0).sum()),supplemental_predictions_matching_4096=int((confirmation[confirmation.n_cells==4096].count_error==0).sum()),supplemental_energy_controlled_cases=int((paired.observed_count_4096>paired.power_lower_count_4096).sum()),supplemental_exact_count_grid_stable=int((paired.observed_count_2048==paired.observed_count_4096).sum()),supplemental_linear_matching=int(sum(r['linear_count']==r['predicted_count'] for r in independent_predictions)),refined_inverse_cases=len(reference),refined_replacement_pairs=len(replacement_pairs),refined_feasible_replacements_2048=int(replacement_pairs.reduced_count_feasible_2048.sum()),refined_feasible_replacements_4096=int(replacement_pairs.reduced_count_feasible_4096.sum()),refined_material_feasibility_changes=int(replacement_pairs.feasibility_changed.sum()),refined_inverse_median_gain_4096_pct=float(inverse[inverse.n_cells==4096].required_qualified_energy_gain_pct.median()),refined_inverse_same_material_feasible_4096=int(inverse[inverse.n_cells==4096].baseline_candidate_energy_feasible.sum()),maximum_refined_replacement_grid_energy_change_pct=float(replacement_pairs.grid_energy_change_pct.abs().max()),publication_status='NOT_PROMOTED',limitations=['Independent output arithmetic and persisted enumeration, not an independent thermal solver or experiments.','Migrated no-loss cases have historical audit fields, not newly recorded raw signed-energy terms.','Stage07 intermediate trajectories do not retain raw pressure and conservation fields; only final pressure and persisted admissibility are available for this arithmetic audit.','Predictions depend on precomputed curves of the same geometries; no unseen-geometry generalization claim.','The original fine-grid reference used binary plus adjacent checks, not global integer enumeration.','Word, final figures, references and complete notebook layout remain to be verified.'])
    cross_material = replacement_pairs[~replacement_pairs.same_material]
    robust = cross_material[cross_material.reduced_count_feasible_2048 & cross_material.reduced_count_feasible_4096]
    summary.update(cross_material_pairs=len(cross_material),cross_material_feasibility_changes=int(cross_material.feasibility_changed.sum()),cross_material_feasible_both_grids=len(robust),same_material_feasibility_changes=int(replacement_pairs[replacement_pairs.same_material].feasibility_changed.sum()),robust_cross_material_energy_gain_min_pct=float(robust.relative_energy_change_pct_4096.min()),robust_cross_material_energy_gain_max_pct=float(robust.relative_energy_change_pct_4096.max()))
    pd.DataFrame(checks).to_csv(OUT/'audit_checks.csv',index=False)
    (OUT/'COMPLETE_ARITHMETIC_AUDIT_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    manifest = {str(p.relative_to(PACKAGE)):sha(p) for p in sorted(DATA.glob('*.csv'))}
    (OUT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if summary['failed_checks']:
        raise RuntimeError('Independent audit failed; do not promote results.')
    return summary


if __name__=='__main__':
    audit()
