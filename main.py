from addon.network import Network, load_network
from addon.decorators import save_tracking_stats

def simulation():
    nodes_to_simulate = -1
    net = Network('test/benchy.gcode', 0.02, 1, node_distance=2/3, extrusion_multiplier=1, saving_path='data/simulation/over')
    net.simulate_printer(node_limit=nodes_to_simulate)
    net.calculate_meshes(processes=None)
    net.save("benchy.pkl")
    save_tracking_stats()

if __name__ =='__main__':
    simulation()

