"""Sampled same-kernel replay with raw pressure/domain/conservation retention.

Selection: first new geometry, uniform particles, both registered confirmation
fluids, 0.1/0.75 MW and 60 h; 4096 cells at selected N and N-1 (eight runs).
This is a boundary audit, not a new predictive test or independent solver.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
import pandas as pd

PACKAGE=Path(__file__).resolve().parents[1]
ROOT=PACKAGE/'upstream_package'
sys.path.insert(0,str(ROOT))
OUT=PACKAGE/'analysis/verified_complete_results/fine_boundary_replay'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def simulate(task):
    from tespub.ltne_v16_signed_inventory import evaluate_packed_bed_v16,BedGeometryV16
    from tespub.fluid_properties import fluid_properties,get_model_spec
    source,config=task['source'],task['config']
    cold,hot=[x+273.15 for x in config['common_duty_C']]
    props=fluid_properties(source['fluid_model_id'],798.15)
    q=task['power_MW']/task['count']
    flow=q*1e6/(props.cp_J_kgK*(hot-cold))
    geometry=BedGeometryV16(**{k:source[k] for k in ['internal_diameter_m','bed_height_m','porosity','particle_diameter_m','topology','gradient_ratio','gradient_orientation','layer_fraction']})
    solid=config['reference_geometry']
    raw=evaluate_packed_bed_v16(model_id=source['fluid_model_id'],design_id=source['design_id'],geometry=geometry,mass_flow_kg_s=flow,
        cold_temperature_K=cold,hot_temperature_K=hot,solid_density_kg_m3=solid['solid_density_kg_m3'],solid_cp_J_kgK=solid['solid_cp_J_kgK'],
        solid_thermal_conductivity_W_mK=solid['solid_thermal_conductivity_W_mK'],n_cells=4096,minimum_outlet_grade_fraction=.9,transport_property_temperature_K=798.15)
    spec=get_model_spec(source['fluid_model_id'])
    domain=raw['temperature_min_K']>=spec.valid_temperature_min_C+273.15-1e-3 and raw['temperature_max_K']<=spec.valid_temperature_max_C+273.15+1e-3
    installed=task['count']*raw['deliverable_energy_MWh']
    admissible=domain and raw['pressure_drop_kPa']<=50 and raw['high_grade_discharge_fraction']>=config['admissibility']['minimum_high_grade_discharge_fraction']
    verified=math.isclose(installed,task['stored_installed_energy_MWh'],rel_tol=1e-12,abs_tol=1e-12) and (admissible and installed+1e-12>=task['required_energy_MWh'])==task['stored_feasible']
    conserved=max(abs(raw['charge_energy_balance_error_pct']),abs(raw['discharge_energy_balance_error_pct']))<=.1
    return dict(design_id=source['design_id'],power_MW=task['power_MW'],count=task['count'],required_energy_MWh=task['required_energy_MWh'],n_cells=4096,installed_energy_MWh=installed,property_domain_supported=domain,matched_stored_boundary=bool(verified),conservation_accepted=bool(conserved),raw=raw)


def main():
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[name]='1'
    OUT.mkdir(exist_ok=True)
    import msvcrt
    lock=(ROOT/'.upgrade_checkpoints_20261006/owner.lock').open('a+b');lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    config=json.loads((ROOT/'config/study_v18.json').read_text())
    source_map=pd.read_csv(ROOT/'results_upgrade_20261006/new_geometry_material_inputs.csv').set_index('design_id').to_dict('index')
    paths=sorted((ROOT/'.upgrade_checkpoints_20261006/07_energy_controlled_fine_grid').glob('*.json'))
    rows=[json.loads(p.read_text())['result'] for p in paths]
    selected=[r for r in rows if r['base_design_id']=='UG0001' and r['topology_id']=='T1_homogeneous_none' and r['n_cells']==4096 and r['energy_to_power_h']==60]
    assert len(selected)==4
    tasks=[]
    for row in selected:
        for entry in row['trajectory'][-2:]:
            source=dict(source_map[row['design_id']],design_id=row['design_id'])
            tasks.append(dict(source=source,config=config,power_MW=row['power_MW'],count=entry['count'],required_energy_MWh=row['required_energy_MWh'],stored_installed_energy_MWh=entry['installed_energy_MWh'],stored_feasible=entry['feasible']))
    identity=digest(dict(script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kernel=hashlib.sha256((ROOT/'tespub/ltne_v16_signed_inventory.py').read_bytes()).hexdigest(),tasks=tasks))
    results,pending=[],[]
    for task in tasks:
        path=OUT/(digest(task)+'.json')
        if path.exists():
            record=json.loads(path.read_text());assert record['identity']==identity and record['digest']==digest(record['result'])
            results.append(record['result'])
        else:pending.append(task)
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures={executor.submit(simulate,task):task for task in pending}
        for future in as_completed(futures):
            result=future.result();task=futures[future];path=OUT/(digest(task)+'.json')
            temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(identity=identity,digest=digest(result),result=result),allow_nan=False,indent=2),encoding='utf-8')
            temporary.replace(path);results.append(result)
            print('Persisted raw boundary replay',len(results),'/8',flush=True)
    summary=dict(identity=identity,checked_at_utc=datetime.now(timezone.utc).isoformat(),sampled_runs=len(results),all_stored_boundaries_matched=all(r['matched_stored_boundary'] for r in results),all_conservation_accepted=all(r['conservation_accepted'] for r in results),all_property_domains_supported=all(r['property_domain_supported'] for r in results),scope='Eight prespecified sampled same-kernel boundary replays, not all 48 confirmations or an independent physical solver.',publication_status='NOT_PROMOTED')
    (OUT/'REPLAY_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lock.close();print(json.dumps(summary,indent=2))
    if not all(summary[k] for k in ['all_stored_boundaries_matched','all_conservation_accepted','all_property_domains_supported']):raise RuntimeError('Raw boundary replay failed')


if __name__=='__main__':main()
