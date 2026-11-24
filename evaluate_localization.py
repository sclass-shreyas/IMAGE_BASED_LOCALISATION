"""Evaluate session logs against simulation truth or an independently recorded CSV.

Reference CSV: timestamp_s,lat,lon[,heading_deg]. Times must share the session
origin. --reference-offset-s is added to reference timestamps explicitly.
No reference is returned to the estimator.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from src.road_network import LocalFrame


def evaluate(path, reference_path=None, offset_s=0, max_gap_s=2):
    records=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    rows=[r for r in records if r.get('type')=='update']
    if not rows:
        raise ValueError('Log has no localization updates')
    first=rows[0]['vehicle']
    frame=LocalFrame(first['lat'],first['lon'])
    reference=[]
    if reference_path:
        with open(reference_path,newline='',encoding='utf-8-sig') as stream:
            reference=[dict(timestamp_s=float(r['timestamp_s'])+offset_s,lat=float(r['lat']),lon=float(r['lon']),
                            heading_deg=float(r['heading_deg']) if r.get('heading_deg') else None) for r in csv.DictReader(stream)]
        reference.sort(key=lambda r:r['timestamp_s'])
    times=np.asarray([r['timestamp_s'] for r in reference])
    errors,baseline_errors,heading_errors,paired=[],[],[],[]
    for row in rows:
        ref=row.get('reference') if not reference_path else None
        t=row['timestamp_s']
        if reference:
            j=int(np.searchsorted(times,t))
            if j<len(times) and abs(times[j]-t)<1e-6:
                ref=reference[j]
            elif 0<j<len(times) and 0<times[j]-times[j-1]<=max_gap_s:
                fraction=(t-times[j-1])/(times[j]-times[j-1])
                ref={k:reference[j-1][k]+fraction*(reference[j][k]-reference[j-1][k]) for k in ('lat','lon')}
        if ref is None:
            continue
        target=frame.xy(ref['lat'],ref['lon'])
        vehicle=row['vehicle'];base=row['baseline']
        err=float(np.linalg.norm(frame.xy(vehicle['lat'],vehicle['lon'])-target))
        base_err=float(np.linalg.norm(frame.xy(base['lat'],base['lon'])-target))
        errors.append(err);baseline_errors.append(base_err)
        if ref.get('heading_deg') is not None:
            heading_errors.append(float(abs((vehicle['heading_deg']-ref['heading_deg']+180)%360-180)))
        paired.append(dict(timestamp_s=t,position_error_m=err,baseline_error_m=base_err))
    valid=[r for r in rows if r['motion'].get('motion_valid')]
    raw_distance=sum(np.hypot(r['motion']['right_delta_m'],r['motion']['forward_delta_m']) for r in valid)
    metrics=dict(updates=len(rows),paired_reference_samples=len(errors),
                 accepted_motion_distance_m=float(raw_distance),
                 rejection_rate=1-len(valid)/len(rows),
                 mean_processing_ms=float(np.mean([r.get('processing_ms',0) for r in rows])),
                 processing_fps=rows[-1].get('processing_fps'),
                 reference_type='external approximate reference' if reference_path else 'simulation truth if present',
                 reference_offset_s=offset_s)
    if errors:
        metrics.update(mean_position_error_m=float(np.mean(errors)),rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
                       baseline_rmse_m=float(np.sqrt(np.mean(np.square(baseline_errors)))),
                       final_drift_m=errors[-1],max_position_error_m=max(errors))
    if heading_errors:
        metrics['mean_absolute_heading_error_deg']=float(np.mean(heading_errors))
    if (rows[-1].get('reference') or {}).get('distance_m') is not None and not reference_path:
        metrics['distance_error_m']=float(raw_distance-rows[-1]['reference']['distance_m'])
    return metrics,paired


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log')
    parser.add_argument('--reference-csv')
    parser.add_argument('--reference-offset-s',type=float,default=0)
    parser.add_argument('--max-reference-gap-s',type=float,default=2)
    parser.add_argument('--output',default='results/evaluation.json')
    parser.add_argument('--measured-distance-m',type=float)
    args=parser.parse_args()
    metrics,pairs=evaluate(args.log,args.reference_csv,args.reference_offset_s,args.max_reference_gap_s)
    if args.measured_distance_m is not None:
        metrics['distance_error_m']=metrics['accepted_motion_distance_m']-args.measured_distance_m
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    if pairs:
        with output.with_suffix('.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
