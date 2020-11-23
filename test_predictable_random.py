import scopesim
import scopesim_templates
import numpy as np


# todo use units here?
cluster = scopesim_templates.basic.stars.cluster(mass=10000, distance=50000, core_radius=2, seed=9001)  # random seed

micado = scopesim.OpticalTrain("MICADO")
def get_image(seed):
    np.random.seed(seed)

    micado.observe(cluster, random_seed=seed, update=True)
    return micado.readout(random_seed=seed)[0][1].data


print(get_image(0) - get_image(1))

