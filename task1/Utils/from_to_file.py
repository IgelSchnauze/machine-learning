def get_mu_sigma(file):
    with open (file, 'r') as f:
        mu = float(f.readline().split()[0])
        sigma = float(f.readline().split()[0])
    return mu, sigma


def write_variables(file, variables):
    with open (file, 'w') as f:
        f.write("\n".join(str(var) for var in variables))