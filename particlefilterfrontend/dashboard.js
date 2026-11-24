/* Default motion dashboard. Reference trajectories never become estimator inputs. */
const $ = id => document.getElementById(id);
const number = id => Number($(id).value);
const checked = id => $(id).checked;
const map = L.map('map', {zoomControl:false, preferCanvas:true}).setView([12.876952,77.53437],16);
L.control.zoom({position:'bottomright'}).addTo(map);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom:19, attribution:'© OpenStreetMap contributors'
}).addTo(map).on('tileerror', () => { $('map-status').textContent = 'Background tiles unavailable; cached road geometry still works.'; });
const pathLine = L.polyline([], {color:'#38bdf8',weight:4}).addTo(map);
const baselineLine = L.polyline([], {color:'#aab3c0',weight:2,dashArray:'5 5'}).addTo(map);
const referenceLine = L.polyline([], {color:'#fb923c',weight:3}).addTo(map);
const plannedLine = L.polyline([], {color:'#c084fc',weight:3,dashArray:'8 5'}).addTo(map);
const gpsLine = L.polyline([], {color:'#fb923c',weight:3,dashArray:'3 3'}).addTo(map);
const particles = L.layerGroup().addTo(map);
let roadLayer, vehicle, uncertainty, headingLine, startMarker, destinationMarker, viaMarker;
let startPosition=null, destination=null, via=null, lastSession=null, lastSequence=-1, paused=false;
let downloadPending=false, referenceSamples=[];

async function api(path, options={}) {
  const response=await fetch('/api/localization'+path, options);
  const data=await response.json();
  if (!response.ok) throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));
  return data;
}
const post=(path, body) => api(path, {method:'POST',headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
const showError=error => { $('error-message').textContent=error.message||String(error); };
function layerVisible(layer, visible) { if (visible && !map.hasLayer(layer)) layer.addTo(map); else if (!visible && map.hasLayer(layer)) map.removeLayer(layer); }
function setPoint(latlng) {
  const mode=$('click-mode').value;
  const p=[latlng.lat,latlng.lng];
  if (mode==='start') {
    startPosition=p;
    if (!startMarker) startMarker=L.marker(p).addTo(map).bindTooltip('Manual start'); else startMarker.setLatLng(p);
    $('start-position').textContent=`Start: ${p.map(v=>v.toFixed(6)).join(', ')}`;
  } else if (mode==='destination') {
    destination=p;
    if (!destinationMarker) destinationMarker=L.marker(p).addTo(map).bindTooltip('Simulation destination'); else destinationMarker.setLatLng(p);
    $('destination-position').textContent=`Destination: ${p.map(v=>v.toFixed(6)).join(', ')}`;
  } else {
    via=p;
    if (!viaMarker) viaMarker=L.marker(p).addTo(map).bindTooltip('Simulation deviation'); else viaMarker.setLatLng(p);
  }
}
map.on('click', e=>setPoint(e.latlng));
$('clear-via').onclick=()=>{via=null;if(viaMarker)map.removeLayer(viaMarker);viaMarker=null;};
async function loadRoads() {
  try {
    const result=await api('/map');
    if (roadLayer) map.removeLayer(roadLayer);
    roadLayer=L.geoJSON(result.geojson,{style:{color:'#9ca3af',weight:3,opacity:.6},onEachFeature:(feature,layer)=>{
      layer.on('click',e=>{
        L.DomEvent.stopPropagation(e);
        setPoint(e.latlng);
        if(checked('align-road') && $('click-mode').value==='start') {
          const pts=feature.geometry.coordinates;
          let best=Infinity,heading=feature.properties.heading_deg;
          for(let i=1;i<pts.length;i++) {
            const a=L.latLng(pts[i-1][1],pts[i-1][0]),b=L.latLng(pts[i][1],pts[i][0]);
            const d=a.distanceTo(e.latlng)+b.distanceTo(e.latlng)-a.distanceTo(b);
            if(d<best) {best=d;heading=(Math.atan2(b.lat-a.lat,(b.lng-a.lng)*Math.cos(a.lat*Math.PI/180))*180/Math.PI+360)%360;}
          }
          $('heading').value=heading.toFixed(1);
        }
      });
    }}).addTo(map);
    $('map-status').textContent=`${result.geojson.features.length} directed road edges · ${result.source}`;
  } catch(error) { $('map-status').textContent=error.message; }
}
function request(preview=false) {
  const kind=$('source-type').value;
  if(!preview && !startPosition) throw new Error('Click the map to select your starting position.');
  return {
    mode:preview?'preview':kind==='simulation'?'simulation':kind==='file'?'file':'live',
    video:{source_type:kind==='simulation'?'webcam':kind,url:$('source-url').value.trim(),
      camera_index:number('camera-index'),width:number('video-width'),height:number('video-height'),fps:number('video-fps')},
    calibration_path:$('calibration-path').value.trim(),lat:startPosition?.[0]??null,lon:startPosition?.[1]??null,
    heading_deg:number('heading'),position_sigma_m:number('position-sigma'),heading_sigma_deg:number('heading-sigma'),
    enable_osm:preview?false:checked('use-osm'),enable_ocr:checked('use-ocr'),destination,via,
    simulation_speed_mps:number('sim-speed'),simulation_noise_m:number('sim-noise'),
    simulation_yaw_drift:number('sim-drift'),playback_rate:number('playback-rate')
  };
}
async function start(preview) {
  try {
    const data=request(preview);
    $('error-message').textContent='';
    await post('/stop');
    await post('/start',data);
  } catch(error) {showError(error);}
}
$('connect-video').onclick=()=>start(true);
$('start-live').onclick=()=>start(false);
$('disconnect-video').onclick=()=>post('/stop').catch(showError);
$('pause-live').onclick=()=>{paused=!paused;post(`/pause?paused=${paused}`).catch(showError);};
$('reset-live').onclick=()=>post('/reset').then(clearTrails).catch(showError);
function clearTrails() {
  for(const line of [pathLine,baselineLine,referenceLine,plannedLine])line.setLatLngs([]);
  particles.clearLayers();
  for(const layer of [vehicle,uncertainty,headingLine])if(layer)map.removeLayer(layer);
  vehicle=uncertainty=headingLine=null;
  lastSequence=-1;
}
$('download-map').onclick=async()=>{
  try {
    const p=map.getCenter();
    await post('/map/download',{lat:p.lat,lon:p.lng,radius_m:number('map-radius')});
    downloadPending=true;
    $('map-status').textContent='Downloading drivable graph in background…';
  } catch(error){showError(error);}
};
$('reference-file').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;
  try {
    const rows=(await file.text()).trim().split(/\r?\n/).map(s=>s.split(',').map(x=>x.trim()));
    const header=rows.shift(), ti=header.indexOf('timestamp_s'),la=header.indexOf('lat'),lo=header.indexOf('lon');
    if(Math.min(ti,la,lo)<0)throw new Error('Reference CSV needs timestamp_s,lat,lon columns.');
    referenceSamples=rows.map(r=>({t:Number(r[ti]),lat:Number(r[la]),lon:Number(r[lo])})).filter(r=>Number.isFinite(r.t)&&Number.isFinite(r.lat)&&Number.isFinite(r.lon)&&Math.abs(r.lat)<=90&&Math.abs(r.lon)<=180).sort((a,b)=>a.t-b.t);
    gpsLine.setLatLngs(referenceSamples.map(r=>[r.lat,r.lon]));
  } catch(error){showError(error);}
};
function trail(line,p) {const points=line.getLatLngs();points.push(p);if(points.length>10000)points.shift();line.setLatLngs(points);}
function fmt(n,d=2){return Number.isFinite(n)?n.toFixed(d):'—';}
async function poll() {
  try {
    const data=await api('/state');
    $('live-status').textContent=(data.status||'idle').toUpperCase();
    paused=data.status==='paused';
    if(data.error)showError(data.error);
    const stream=data.stream||{};
    $('stream-info').textContent=`${stream.status||'disconnected'} · ${stream.width||'—'}×${stream.height||'—'} · skipped ${stream.dropped_frames??0} · host queue age ${fmt(stream.receive_to_process_ms,0)} ms · sensor latency unknown`;
    if(data.session_id && data.session_id!==lastSession){clearTrails();lastSession=data.session_id;}
    if(stream.width && ['running','paused','completed','stopped'].includes(data.status)) {
      $('camera-feed').hidden=false;$('camera-placeholder').hidden=true;
      $('camera-feed').src=`/api/localization/preview.jpg?bird=${checked('show-bird')}&t=${Date.now()}`;
    } else { $('camera-feed').hidden=true;$('camera-placeholder').hidden=false; }
    if(data.log_path)$('log-path').textContent=`Session log: ${data.log_path}`;
    if(data.planned_route)plannedLine.setLatLngs(data.planned_route);
    if(data.vehicle) {
      const v=data.vehicle,p=[v.lat,v.lon],h=v.heading_deg*Math.PI/180;
      $('coord-display').textContent=`${v.lat.toFixed(6)}, ${v.lon.toFixed(6)}\nHeading ${fmt(v.heading_deg,1)}° from east · compass ${fmt((90-v.heading_deg+360)%360,1)}°`;
      $('uncertainty-display').textContent=`${fmt(v.uncertainty_meters)} m spread · ${fmt(v.radius95_m)} m radius\nHeading σ ${fmt(v.heading_sigma_deg,1)}°`;
      const m=data.motion||{};
      $('motion-display').textContent=`${m.status||'initialized'} · features ${m.tracked_features??0} · inliers ${fmt(m.inlier_ratio)}\nForward ${fmt(m.forward_delta_m,3)} m · right ${fmt(m.right_delta_m,3)} m · yaw ${fmt(m.yaw_delta_rad,3)} rad\nResidual ${fmt(m.tracking_error,3)} m · FB ${fmt(m.fb_error_px)} px`;
      const experimentalDistance = Number.isFinite(data.distance_m) ? data.distance_m * 2.4 : null;
      $('distance-display').textContent=`${fmt(data.speed_mps)} m/s · Raw distance: ${fmt(data.distance_m)} m\nExperimental ×2.4: ${fmt(experimentalDistance)} m\nUnvalidated comparison only — not used by filter\nRejected ${(100*(data.rejection_rate||0)).toFixed(1)}%`;
      $('filter-display').textContent=`${data.particle_count} particles · ESS ${fmt(data.effective_sample_size,0)}\n${data.diagnostics?.road_status||'—'} · ${data.road_hypotheses?.length??0} road hypotheses`;
      $('performance-display').textContent=`${fmt(data.processing_fps,1)} updates/s · ${fmt(data.processing_ms,1)} ms VO\n${fmt(data.timestamp_s)} s`;
      if(!vehicle){vehicle=L.circleMarker(p,{radius:7,color:'#38bdf8',fillOpacity:1}).addTo(map);uncertainty=L.circle(p,{color:'#38bdf8',fillOpacity:.12,weight:1}).addTo(map);headingLine=L.polyline([],{color:'#38bdf8',weight:4}).addTo(map);}
      vehicle.setLatLng(p);uncertainty.setLatLng(p).setRadius(v.radius95_m);
      headingLine.setLatLngs([p,[v.lat+15*Math.sin(h)/111000,v.lon+15*Math.cos(h)/(111000*Math.cos(v.lat*Math.PI/180))]]);
      if(data.sequence!==lastSequence && data.sequence>0) {
        lastSequence=data.sequence;trail(pathLine,p);
        if(data.baseline)trail(baselineLine,[data.baseline.lat,data.baseline.lon]);
        if(data.reference)trail(referenceLine,[data.reference.lat,data.reference.lon]);
        particles.clearLayers();for(const xy of data.particles||[])L.circleMarker(xy,{radius:2,stroke:false,fillColor:'#fde047',fillOpacity:.4}).addTo(particles);
        if(checked('follow-map'))map.panTo(p,{animate:false});
      }
      $('reference-error').textContent=Number.isFinite(data.position_error_m)?`Simulated truth: PF ${fmt(data.position_error_m)} m · VO ${fmt(data.baseline_error_m)} m`:'No timestamp-aligned reference';
      if(referenceSamples.length && Number.isFinite(data.timestamp_s)) {
        let nearest=referenceSamples.reduce((a,b)=>Math.abs(a.t-data.timestamp_s)<Math.abs(b.t-data.timestamp_s)?a:b);
        if(Math.abs(nearest.t-data.timestamp_s)<=1)$('reference-error').textContent=`Approximate reference error: ${fmt(map.distance(p,[nearest.lat,nearest.lon]))} m (sample within 1 s)`;
      }
    }
    layerVisible(particles,checked('show-particles'));layerVisible(baselineLine,checked('show-baseline'));
    layerVisible(referenceLine,checked('show-reference'));layerVisible(gpsLine,checked('show-reference'));layerVisible(plannedLine,checked('show-planned'));
    if(downloadPending){const d=await api('/map/download');if(d.status!=='downloading'){downloadPending=false;if(d.status==='complete')await loadRoads();else showError(d.error);}}
  } catch(error) {$('live-status').textContent='API OFFLINE';}
  setTimeout(poll,500);
}

// Preserve the existing draggable, manually resizable camera panel.
const preview=$('camera-preview'),header=$('camera-preview-header'),panel=$('map-panel');
let drag=null;
header.addEventListener('pointerdown',e=>{
  if(e.button!==0)return;const r=preview.getBoundingClientRect(),p=panel.getBoundingClientRect();
  preview.style.left=`${r.left-p.left}px`;preview.style.top=`${r.top-p.top}px`;preview.style.right='auto';
  drag={id:e.pointerId,x:e.clientX-r.left,y:e.clientY-r.top};header.setPointerCapture(e.pointerId);
});
header.addEventListener('pointermove',e=>{
  if(!drag||e.pointerId!==drag.id)return;const r=panel.getBoundingClientRect();
  preview.style.left=`${Math.max(0,Math.min(panel.clientWidth-preview.offsetWidth,e.clientX-r.left-drag.x))}px`;
  preview.style.top=`${Math.max(0,Math.min(panel.clientHeight-preview.offsetHeight,e.clientY-r.top-drag.y))}px`;
});
header.addEventListener('pointerup',()=>drag=null);header.addEventListener('pointercancel',()=>drag=null);
api('/config').then(c=>{
  map.setView(c.area_center,16);$('source-type').value=c.video.source_type;$('source-url').value=c.video.url;
  $('camera-index').value=c.video.camera_index;$('video-width').value=c.video.width;$('video-height').value=c.video.height;
  $('video-fps').value=c.video.fps;$('calibration-path').value=c.calibration_path;$('map-radius').value=c.area_radius_m;
  $('use-ocr').checked=c.enable_ocr;
  if(c.initial_position)setPoint(L.latLng(...c.initial_position));
  if(c.initial_heading_deg!==null)$('heading').value=c.initial_heading_deg;
  loadRoads();
}).catch(showError);
poll();
