import dataset
import loss 

def train_network(model, network, dataloaders_list, optimizer,
                  number_of_epochs, batch_size, hyperparam_dict):
    """ Training the network """
    loader_interior = dataloaders_list[0]
    loader_boundary = dataloaders_list[1]
    loader_initial = dataloaders_list[2]
    loader_meas = dataloaders_list[3]
    lambda_res = hyperparam_dict["lambda_res"]
    lambda_Lp = hyperparam_dict["lambda_Lp"]
    p_exponent = hyperparam_dict["p_exponent"]

    # Test dataset 
    test_domain = model.domain
    test_domain[0] = torch.tensor([T, T])
    test_data = dataset.assemble_dataset_interior(test_domain, 100) 
    test_data = torch.cat([test_data, model.exact_solution(test_data)], dim=1)
    # ------------

    history = []
    test_error_history = []
    for epoch in number_of_epochs:
        epoch_loss = 0.
        count = 0
        test_error_history.append(loss.relative_test_error(data_test, network))
        for interior_batch,
            boundary_batch, 
            initial_batch,
            meas_batch in zip(loader_interior, loader_boundary, loader_initial, loader_meas):

                loss = Loss(network, model)
                
                optimizer.zero_grad()

                pde_loss = loss.get_pde_loss(interior_batch)
                boundary_loss = loss.get_boundary_condition_loss(boundary_batch)
                initial_loss = loss.get_initial_condition_loss(initial_batch)
                meas_loss = loss.get_measurements_loss(meas_batch)
                reg_loss = loss.get_regularization_loss(p_exponent)

                batch_loss = lambda_res * pde_loss +
                             boundary_loss +
                             meas_loss + 
                             lambda_Lp * reg_loss

                epoch_loss += batch_loss
                count += 1
                batch_loss.backward()
                optimizer.step()

        history.append(epoch_loss / count)

    return np.array(history), np.array(test_error_history)

                

