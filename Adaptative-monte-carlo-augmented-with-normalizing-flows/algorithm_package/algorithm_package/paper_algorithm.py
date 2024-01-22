from numpy.random import uniform
import torch
import torch.autograd as autograd
from  torch.distributions import multivariate_normal
from tqdm import tqdm
import torch.optim as optim
from os import getcwd, makedirs

def NF_MCMC_algorithm(model_name, beta, U, BC, energy_parameters, flow, initial_data, base_distribution, time_step, k_max, k_lang, epsilon):
    #all operations must act on the whole array of lattices, hence proposed_model_configuration_amongst_zeros and other weird stuff
    
    path = getcwd()+'\\saved_models\\'+model_name
    makedirs(path)

    def gradU(configuration):
        return autograd.grad(U(configuration, energy_parameters), configuration)[0]

    n, N = initial_data.shape

    normal_distribution_for_langevin = multivariate_normal.MultivariateNormal(loc=torch.zeros(N), covariance_matrix=torch.eye(N))

    array_of_model_configurations = torch.zeros(k_max,n,N)
    array_of_model_configurations[0] = initial_data
    array_of_model_configurations.requires_grad = True

    optimizer = optim.Adam(flow.parameters(), lr=epsilon)

    history = torch.zeros(k_max, n)
    #history[k,i] = 0 => Dynamic Langevin step
    #history[k,i] = 1 => Static Langevin step
    #history[k,i] = 2 => Dynamic Flow step
    #history[k,i] = 3 => Static Flow step

    for k in tqdm(range(1,k_max)):
        for i in range(n):
            proposed_model_configuration_amongst_zeros = torch.zeros(k_max,n,N)
            if k % k_lang == 0:
                history[k,i] += 2
                proposed_model_configuration_amongst_zeros[k,i] = flow(base_distribution.sample())[0]
                
                proposed_model_configuration_amongst_zeros[k,i] = BC(proposed_model_configuration_amongst_zeros[k,i])

                acceptance_rate = torch.exp(log_rho_hat(array_of_model_configurations[k-1,i])
                                            - log_rho_hat(proposed_model_configuration_amongst_zeros[k,i])
                                            + beta*U(array_of_model_configurations[k-1, i], energy_parameters)
                                            - beta*U(proposed_model_configuration_amongst_zeros[k, i], energy_parameters))

            else:

                proposed_model_configuration_amongst_zeros[k,i] = ( array_of_model_configurations[k-1,i]
                                                                    - time_step * gradU(array_of_model_configurations[k-1, i])
                                                                    + torch.sqrt(2*torch.tensor(time_step)) * normal_distribution_for_langevin.sample())

                proposed_model_configuration_amongst_zeros[k,i] = BC(proposed_model_configuration_amongst_zeros[k,i])

                acceptance_rate = torch.exp(beta*U(array_of_model_configurations[k-1, i], energy_parameters)
                                            - beta*U(proposed_model_configuration_amongst_zeros[k, i], energy_parameters))

            if uniform() > acceptance_rate:
                    proposed_model_configuration_amongst_zeros[k,i] = array_of_model_configurations[k-1,i]
                    history[k,i] += 1
            array_of_model_configurations = array_of_model_configurations + proposed_model_configuration_amongst_zeros
        
        def log_rho_hat(x):
            return base_distribution.log_prob(BC((flow.inverse(x))[0]))+flow.inverse(x)[1]
        
        #OPTIMISATION
        optimizer.zero_grad()
        x = array_of_model_configurations[k-1,:].clone().detach().requires_grad_(False)
        loss = - (base_distribution.log_prob(BC(flow.inverse(x)[0])) + flow.inverse(x)[1]).mean()
        loss.backward()
        optimizer.step()
    
    torch.save(flow.state_dict(), path+'\\model.pt')
    torch.save(array_of_model_configurations, path+'\\array_of_model_configurations.pt')
    torch.save(history, path+'\\history.pt')

    file_name = path+"\\parameters.txt"
    with open(file_name, 'w') as file:
        file.write(f"beta\t{beta}\n")
        file.write(f"n\t{n}\n")
        file.write(f"N\t{N}\n")
        file.write(f"time_step\t{time_step}\n")
        file.write(f"k_max\t{k_max}\n")
        file.write(f"k_lang\t{k_lang}\n")
        file.write(f"epsilon\t{epsilon}\n")
        file.write("energy_parameters\t")
        file.write(str(energy_parameters))
    
    return history, array_of_model_configurations.detach()