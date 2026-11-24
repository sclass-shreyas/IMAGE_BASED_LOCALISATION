"""Replay recorded metric packets with the same filter, optionally disabling OSM."""
import argparse
import json
from pathlib import Path
from particlefilterbackend.road_filter import RoadParticleFilter, FilterSettings
from src.road_network import RoadNetwork, load_graph
from src.osm_matching import config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log')
    parser.add_argument('--disable-osm',action='store_true')
    parser.add_argument('--output',default='results/replayed_motion.jsonl')
    args=parser.parse_args()
    rows=[json.loads(line) for line in Path(args.log).read_text(encoding='utf-8').splitlines() if line.strip()]
    metadata=next(r['config'] for r in rows if r['type']=='metadata')
    lat,lon,heading=metadata['initial_pose']
    area=metadata.get('map_area',{})
    center=area.get('center',config.AREA_CENTER)
    radius=area.get('radius_m',config.AREA_RADIUS_M)
    network=None if args.disable_osm else RoadNetwork(load_graph(center,radius),center)
    pf=RoadParticleFilter(lat,lon,heading,metadata.get('position_sigma_m',3),metadata.get('heading_sigma_deg',10),
                          network,not args.disable_osm,FilterSettings(**metadata.get('filter_settings',{})),metadata.get('seed',0))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    metadata={**metadata,'enable_osm':not args.disable_osm,'replay_source':str(args.log)}
    with output.open('w',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(type='metadata',config=metadata))+'\n')
        for row in rows:
            if row['type']!='update':
                continue
            result=pf.step(row['motion'])
            result.pop('particles',None)
            # Evaluation fields are copied only after the estimator update.
            result.update(type='update',timestamp_s=row['timestamp_s'],reference=row.get('reference'),
                          processing_ms=0,processing_fps=None)
            stream.write(json.dumps(result)+'\n')
    print(output)


if __name__=='__main__':
    main()
