from task1.Operations.get_variables import get_stochastic_var
from task1.Utils.from_to_file import get_mu_sigma, write_variables

if __name__ == '__main__':
    file_from = 'Utils/data.txt'
    file_to = 'Utils/result.txt'

    mu, sigma = get_mu_sigma(file_from)
    n = 1000
    variables = get_stochastic_var(mu, sigma, n)
    write_variables(file_to, variables)