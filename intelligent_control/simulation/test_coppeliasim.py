from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')
clip_angle = lambda x: min(max(x, 10), 170)

sim.clearStringSignal("motorCtrl")

sim.startSimulation()
motor_control = f"0:0,1:90,2:90,3:90"
sim.setStringSignal("motorCtrl", motor_control)
while (t := sim.getSimulationTime()) < 15:
    if t > 7:
        sim.setStringSignal("motorCtrl", "0:180")
    print(f'Simulation time: {t:.2f} [s]')
sim.stopSimulation()