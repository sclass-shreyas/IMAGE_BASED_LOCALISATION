import numpy as np
import pytest
import networkx as nx
from shapely.geometry import LineString
from src.ground_calibration import fit_calibration, project
from visual_odometry import fit_rigid_motion, VOSettings, GroundPlaneVisualOdometry
from src.road_network import RoadNetwork
from particlefilterbackend.road_filter import RoadParticleFilter, advance_pose


def small_network():
    g = nx.MultiDiGraph(crs='EPSG:4326')
    for node, xy in {0:(77., 12.), 1:(77.001, 12.), 2:(77.002, 12.), 3:(77.001, 12.001),
                     4:(77., 12.0003), 5:(77.002, 12.0003)}.items():
        g.add_node(node, x=xy[0], y=xy[1])
    for a,b in [(0,1),(1,2),(1,3),(4,5)]:
        xy = [(g.nodes[n]['x'],g.nodes[n]['y']) for n in (a,b)]
        g.add_edge(a,b,geometry=LineString(xy))
    return RoadNetwork(g,(12.,77.))


def calibration():
    # Synthetic planar camera used only as a mathematical fixture.
    pixels = [[50,350],[590,350],[50,470],[590,470],[300,400],[150,430]]
    matrix = np.array([[.01,0,-3.2],[0,-.01,5],[0,0,1.]])
    return dict(image_width=640,image_height=480,points=[
        dict(pixel_xy=p,ground_right_forward_m=project([p],matrix)[0].tolist()) for p in pixels])


def test_calibration_holdout_and_degenerate():
    source=calibration()
    source['points'][-1]['validation']=True
    result=fit_calibration(source)
    assert max(result['validation_residuals_m']) < 1e-5
    source['points']=[dict(pixel_xy=[x,x],ground_right_forward_m=[x,x]) for x in range(6)]
    with pytest.raises(ValueError,match='collinear'):
        fit_calibration(source)


def test_rigid_camera_motion_sign_and_outliers():
    rng=np.random.default_rng(42)
    old=rng.uniform([-3,1],[3,12],(120,2))
    yaw=.1
    rot=np.array([[np.cos(-yaw),-np.sin(-yaw)],[np.sin(-yaw),np.cos(-yaw)]])
    translation=np.array([.02,.3])
    new=(old-translation)@rot.T
    new[:20]=rng.normal(0,20,(20,2))
    delta, recovered, keep, residual, scale=fit_rigid_motion(old,new,VOSettings())
    np.testing.assert_allclose(delta,translation,atol=1e-5)
    assert recovered==pytest.approx(yaw,abs=1e-5)
    assert keep.sum()>=95


def test_local_frame_composition_across_turn():
    pose=np.zeros(3)
    advance_pose(pose,dict(right_delta_m=0,forward_delta_m=10,yaw_delta_rad=np.pi/2))
    advance_pose(pose,dict(right_delta_m=0,forward_delta_m=5,yaw_delta_rad=0))
    np.testing.assert_allclose(pose,[10,5,np.pi/2],atol=1e-8)


def test_road_heading_and_disconnected_transition():
    net=small_network()
    xy=net.frame.xy([12.,12.,12.0006],[77.0005]*3)
    match=net.match(xy,np.array([0,np.pi,0]))
    assert match['loglike'][0]>match['loglike'][1]
    # First and parallel road do not share nodes, despite spatial proximity.
    penalties=net.transition_penalty(np.array([0]),np.array([3]),np.array([50.]),np.array([50.]),2)
    assert penalties[0]<.1


def test_filter_missing_motion_expands_uncertainty_and_rejects_duplicate_time():
    pf=RoadParticleFilter(12.,77.,0,position_sigma_m=.1,heading_sigma_deg=0,enable_osm=False)
    before=pf.state()['vehicle']['uncertainty_meters']
    pf.step(dict(timestamp_s=1,delta_time_s=1,motion_valid=False))
    assert pf.state()['vehicle']['uncertainty_meters']>before+1
    with pytest.raises(ValueError,match='increase'):
        pf.step(dict(timestamp_s=1,motion_valid=False))


def test_vo_rejects_resolution_blur_and_gaps():
    vo=GroundPlaneVisualOdometry(fit_calibration(calibration()))
    frame=np.zeros((480,640,3),np.uint8)
    assert not vo.process_frame(frame,0)['motion_valid']
    assert vo.process_frame(frame,.1)['status']=='blurred_frame'
    assert vo.process_frame(frame,2)['status']=='invalid_frame_interval'
    with pytest.raises(ValueError,match='resolution'):
        vo.process_frame(np.zeros((240,320,3),np.uint8),3)


def test_vo_on_synthetic_translated_ground_texture():
    import cv2
    rng=np.random.default_rng(3)
    image=rng.integers(0,256,(480,640,3),dtype=np.uint8)
    image=cv2.GaussianBlur(image,(3,3),0)
    vo=GroundPlaneVisualOdometry(fit_calibration(calibration()))
    vo.process_frame(image,0)
    # Image moves down 3 px: plane features move -0.03 m forward,
    # hence camera moves +0.03 m forward. This is synthetic, not a field test.
    shifted=cv2.warpAffine(image,np.float32([[1,0,0],[0,1,3]]),(640,480))
    result=vo.process_frame(shifted,.1)
    assert result['motion_valid'],result
    assert result['forward_delta_m']==pytest.approx(.03,abs=.004)
    assert abs(result['right_delta_m'])<.004


def test_filter_soft_road_constraint_keeps_connected_turn_possible():
    net=small_network()
    pf=RoadParticleFilter(12.,77.0005,0,network=net)
    for i in range(15):
        pf.step(dict(timestamp_s=i+1,delta_time_s=1,motion_valid=True,confidence=1,
                     forward_delta_m=2,right_delta_m=0,yaw_delta_rad=0))
    state=pf.state()
    assert state['diagnostics']['road_status']=='weighted'
    assert np.isfinite(pf.weights).all() and pf.weights.sum()==pytest.approx(1)
    assert state['distance_m']==pytest.approx(30)
