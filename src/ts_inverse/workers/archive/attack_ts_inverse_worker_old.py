
import numpy as np
import torch
import torch.nn.functional as F


from src.models import get_fc_state_dict
from src.attack_time_series_utils import interpolate
from src.workers.attack_dlg_invg_dia_worker import AttackBaselineWorker


class AttackTSInverseWorker(AttackBaselineWorker):

    def __init__(self, worker_id):
        worker_name = 'AttackTSInverseWorker'
        super().__init__(worker_id, worker_name)

    def worker_process(self, c, d_c, m_c, fam_c):
        model, train_dataloader, final_config = self._init_attack_worker_process(c, d_c, m_c, fam_c)
        (dummy_inputs, dummy_targets), (inputs, targets) = self.perform_TS_Inverse_attack(model, train_dataloader, config=final_config)
        return dummy_inputs, dummy_targets, inputs, targets

    def init_logger_object(self, config):
        tags = ['TS-Inverse']
        project_names = {
            'wandb': 'thesis-ts-inverse',
            'comet_ml': 'tno-fl-dlg',
            'neptune': 'tno-fl-dlg/forecasting',
            'clear_ml': 'thesis-forecasting'
        }
        return self._init_logger_object(project_names, tags, config)

    def perform_TS_Inverse_attack(self, model, tr_dataloader, config):
        logger_object_dict = self.init_logger_object(config)
        number_of_batches = config['number_of_batches']
        learning_rate = config['learning_rate']
        num_attack_steps = config['num_attack_steps']
        device = config['device']
        verbose = config['verbose']

        ### RESOLUTION WARPING ###
        if config['resolution_warp_settings']:
            resolution_warp_settings = config['resolution_warp_settings']
            resolution_warp_factor = resolution_warp_settings['resolution_warp_factor']
            warp_style = resolution_warp_settings['resolution_warping_style']
            if warp_style == 'original-to-small':
                warp_n_times = resolution_warp_settings['warp_n_times']
                warp_limit_factor = resolution_warp_settings['warp_limit_factor']
            if resolution_warp_factor == 1:
                warp_style = 'none'
        else:
            warp_style = 'none'
        ##########################

        ###### CLAMP #######
        clamp_every_n_steps = config['clamp_dummy_data_every_n_steps']
        ####################

        ##### TARGETS FIRST #####
        target_optimization_first_factor = config['target_optimization_first_factor']
        #########################

        #### RNN ATTACK ####
        rnn_attack = config['rnn_attack']
        rnn_attack_epsilon = 0.0001
        ####################

        #### Gradient loss ####
        gradient_loss = config['gradient_loss']
        #######################

        all_batch_inputs, all_batch_targets, all_model_state_dicts, all_model_gradients, all_model_updates = self.train_model_and_record(model, tr_dataloader, config)

        if rnn_attack:
            gru_hidden_state_shape = None 
            for key, value in all_model_state_dicts[0].items():
                if 'weight_hh' in key:
                    gru_hidden_state_shape = torch.Size((all_batch_targets[0].shape[0], value.shape[1]))
                    break

            all_batch_hidden_states = [torch.zeros(gru_hidden_state_shape, device=device, requires_grad=True) for _ in range(number_of_batches)]

            all_dummy_hidden_states = [torch.rand(gru_hidden_state_shape, device=device, requires_grad=True) for _ in range(number_of_batches)]
            all_dummy_targets = [torch.rand_like(all_batch_targets[0], device=device, requires_grad=True) for _ in range(number_of_batches)]

            linear_model = torch.nn.Linear(all_model_state_dicts[0]['fc.weight'].shape[1], all_model_state_dicts[0]['fc.weight'].shape[0]).to(device)

            for batch_number in range(number_of_batches):
                linear_model.load_state_dict(get_fc_state_dict(all_model_state_dicts[batch_number]))
                model.load_state_dict(all_model_state_dicts[batch_number])
                original_dy_dx = all_model_gradients[batch_number][-2:]
                dummy_hidden_states = all_dummy_hidden_states[batch_number]
                dummy_targets = all_dummy_targets[batch_number]
                batch_hidden_states = all_batch_hidden_states[batch_number]
                batch_targets = all_batch_targets[batch_number]
                sample_mapping = np.arange(0, batch_targets.shape[0])

                dummy_optimizer = torch.optim.Adam([dummy_hidden_states, dummy_targets], lr=learning_rate)

                for attack_step in range(0, num_attack_steps+1):
                    attack_metrics = {
                        'batch_number': batch_number,
                        'step': attack_step,
                        'sample_mapping': sample_mapping,
                    }
                    dummy_optimizer.zero_grad()
                    dummy_out = linear_model(dummy_hidden_states)
                    dummy_y = F.mse_loss(dummy_out, dummy_targets)
                    dummy_dy_dx = torch.autograd.grad(dummy_y, linear_model.parameters(), create_graph=True)
                    dy_dx_loss = self.gradient_loss_function(dummy_dy_dx, original_dy_dx, gradient_loss)
                    dy_dx_loss.backward()
                    dummy_optimizer.step()

                    ###### CLAMP #######
                    if clamp_every_n_steps > 0 and attack_step % clamp_every_n_steps == 0:
                        dummy_targets.data.clamp_(min=0)
                    ####################
                    
                    # Should calcualte evalaution metrics and log them
                    if attack_step % (num_attack_steps // min(num_attack_steps, 500)) == 0:
                        self.evaluate_and_log_reconstruction(logger_object_dict, num_attack_steps, verbose, batch_number,
                                                            batch_hidden_states, batch_targets, attack_step, attack_metrics, dummy_hidden_states, dummy_targets, dy_dx_loss)
            self._end_logger_object(logger_object_dict)
            return (all_dummy_hidden_states, all_dummy_targets), (all_batch_inputs, all_batch_targets)
        else:
            all_dummy_inputs, all_dummy_targets, warped_input_shape, warped_target_shape = self.generate_dummy_data(all_batch_inputs[0], all_batch_targets[0], config)
            for batch_number in range(number_of_batches):
                model.load_state_dict(all_model_state_dicts[batch_number])
                original_dy_dx = all_model_gradients[batch_number]
                dummy_inputs = all_dummy_inputs[batch_number]
                dummy_targets = all_dummy_targets[batch_number]
                batch_inputs = all_batch_inputs[batch_number]
                batch_targets = all_batch_targets[batch_number]
                sample_mapping = np.arange(0, batch_inputs.shape[0])

                ##### TARGETS FIRST #####
                if target_optimization_first_factor > 0 or rnn_attack:
                    dummy_targets_optimizer = torch.optim.Adam([dummy_targets], lr=learning_rate)
                dummy_optimizer = torch.optim.Adam([dummy_inputs, dummy_targets], lr=learning_rate)
                #########################

                for attack_step in range(0, num_attack_steps+1):
                    attack_metrics = {
                        'batch_number': batch_number,
                        'step': attack_step,
                        'sample_mapping': sample_mapping,
                    }

                    ##### TARGETS FIRST #####
                    if target_optimization_first_factor > 0 and attack_step < num_attack_steps*target_optimization_first_factor or rnn_attack:
                        dummy_targets_optimizer.zero_grad()
                    else:
                        dummy_optimizer.zero_grad()
                    #########################

                    ### RESOLUTION WARPING ###
                    if warp_style == 'small-to-original':
                        modified_inputs = interpolate(dummy_inputs, batch_inputs.shape[1])
                    elif warp_style == 'original-to-small' and (attack_step % int(num_attack_steps*warp_limit_factor // warp_n_times) == 0) and \
                            attack_step < num_attack_steps*warp_limit_factor:
                        modified_inputs = interpolate(dummy_inputs, warped_input_shape[1])  # Scale down
                        modified_inputs = interpolate(modified_inputs, batch_inputs.shape[1])  # Scale up & smooth
                        dummy_inputs.data = modified_inputs.data
                    ##########################
                    else:
                        modified_inputs = dummy_inputs

                    dummy_out = model(modified_inputs)

                    ### RESOLUTION WARPING ###
                    if warp_style == 'small-to-original':
                        modified_targets = interpolate(dummy_targets.unsqueeze(-1), batch_targets.shape[1]).squeeze(-1)
                    elif warp_style == 'original-to-small' and ((attack_step+1) % int((num_attack_steps*warp_limit_factor) // warp_n_times) == 0) and \
                            attack_step <= num_attack_steps*warp_limit_factor+1:
                        modified_targets = interpolate(dummy_targets.unsqueeze(-1), warped_target_shape[1])  # Scale down
                        modified_targets = interpolate(modified_targets, batch_targets.shape[1]).squeeze(-1)  # Scale up & smooth
                        dummy_targets.data = modified_targets.data
                    ##########################
                    else:
                        modified_targets = dummy_targets

                    dummy_y = F.mse_loss(dummy_out, modified_targets)
                    dummy_dy_dx = torch.autograd.grad(dummy_y, model.parameters(), create_graph=True)
                    dy_dx_loss = self.gradient_loss_function(dummy_dy_dx, original_dy_dx, gradient_loss)
                    dy_dx_loss.backward()

                    # # Tries to increase the gradients the more in the past they are in an attempt to mitigate the vanishing gradient problem
                    if rnn_attack and (model.name == 'GRU_Predictor' or model.name == 'JitGRU_Predictor'):
                        grad_sign = dummy_inputs.grad.sign() * (dummy_inputs.grad.abs() > 0)
                        dummy_inputs.data -= rnn_attack_epsilon * grad_sign
                        dummy_targets_optimizer.step()
                    else:

                        ##### TARGETS FIRST #####
                        if target_optimization_first_factor > 0 and attack_step < num_attack_steps*target_optimization_first_factor:
                            dummy_targets_optimizer.step()
                        else:
                            dummy_optimizer.step()
                        #########################

                    ###### CLAMP #######
                    if clamp_every_n_steps > 0 and attack_step % clamp_every_n_steps == 0:
                        dummy_inputs.data.clamp_(min=0)
                        dummy_targets.data.clamp_(min=0)
                    ####################

                    # Should calcualte evalaution metrics and log them
                    if attack_step % (num_attack_steps // min(num_attack_steps, 500)) == 0:
                        self.evaluate_and_log_reconstruction(logger_object_dict, num_attack_steps, verbose, batch_number,
                                                            batch_inputs, batch_targets, attack_step, attack_metrics, modified_inputs, modified_targets, dy_dx_loss)

        model.eval()
        self._end_logger_object(logger_object_dict)
        return (dummy_inputs, dummy_targets), (all_batch_inputs, all_batch_targets)



    def train_model_and_record(self, model, tr_dataloader, config):
        learning_rate, device, warmup_number_of_batches, number_of_batches = config['learning_rate'], config['device'], config['warmup_number_of_batches'], config['number_of_batches']

        model.to(device)
        # model.eval()
        model_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        all_batch_inputs, all_batch_targets, all_model_state_dicts, all_model_gradients, all_model_updates = [], [], [], [], []
        for batch_number, (batch_inputs, batch_targets) in enumerate(tr_dataloader):
            if batch_number >= warmup_number_of_batches+number_of_batches:
                break

            batch_inputs, batch_targets = batch_inputs[:, :, model.features].to(device), batch_targets[:, :, 0].to(device)
            all_batch_inputs.append(batch_inputs.clone())
            all_batch_targets.append(batch_targets.clone())
            model_optimizer.zero_grad()
            if batch_number >= warmup_number_of_batches:
                all_model_state_dicts.append(model.state_dict().copy())  # Save the model parameters
            out = model(batch_inputs)
            y = F.mse_loss(out, batch_targets)
            y.backward()
            if batch_number >= warmup_number_of_batches:
                all_model_gradients.append([param.grad.detach().clone() for param in model.parameters()])  # Save the model gradients
            model_optimizer.step()
            if batch_number >= warmup_number_of_batches:
                model_update = [(current - prev).detach().clone() for prev, current in zip(all_model_state_dicts[-1].values(), model.state_dict().values())]
                all_model_updates.append(model_update)

        return all_batch_inputs, all_batch_targets, all_model_state_dicts, all_model_gradients, all_model_updates

    def generate_dummy_data(self, batch_input_example, batch_target_example, config):
        number_of_batches, resolution_warp_settings, device = config['number_of_batches'], config['resolution_warp_settings'], config['device']
        ### RESOLUTION WARPING ###
        warped_input_shape = batch_input_example.shape
        warped_target_shape = batch_target_example.shape
        if resolution_warp_settings and resolution_warp_settings['warp_style'] != 'none':
            warped_input_shape = (
                batch_input_example.shape[0],
                int(batch_input_example.shape[1] // resolution_warp_settings['resolution_warp_factor']),
                batch_input_example.shape[2])
            warped_target_shape = (batch_target_example.shape[0], int(batch_target_example.shape[1]//resolution_warp_settings['resolution_warp_factor']))
        if resolution_warp_settings and resolution_warp_settings['warp_style'] == 'small-to-original':
            all_dummy_inputs = [torch.rand(warped_input_shape, device=device, requires_grad=True) for _ in range(number_of_batches)]
            all_dummy_targets = [torch.rand(warped_target_shape, device=device, requires_grad=True) for _ in range(number_of_batches)]
            return all_dummy_inputs, all_dummy_targets,
        else:  # Even if large to small, it needs to be batch size
            all_dummy_inputs = [torch.rand_like(batch_input_example, device=device, requires_grad=True) for _ in range(number_of_batches)]
            all_dummy_targets = [torch.rand_like(batch_target_example, device=device, requires_grad=True) for _ in range(number_of_batches)]
        ##########################

        return all_dummy_inputs, all_dummy_targets, warped_input_shape, warped_target_shape

    def evaluate_and_log_reconstruction(
            self, logger_object_dict, num_attack_steps, verbose, batch_number, batch_inputs, batch_targets, attack_step, attack_metrics, modified_inputs,
            modified_targets, dy_dx_loss):
        with torch.no_grad():
            attack_metrics['grad_diff_loss_mse'] = dy_dx_loss.detach().item()

            # sample_mapping = get_batch_sample_mapping(batch_inputs, modified_inputs)
            sample_mapping = self.get_batch_sample_mapping(batch_targets, modified_targets)

            mean_evaluation = {
                'inputs/mse/mean': F.mse_loss(modified_inputs[sample_mapping], batch_inputs).item(),
                'targets/mse/mean': F.mse_loss(modified_targets[sample_mapping], batch_targets).item(),
                'inputs/rmse/mean': torch.sqrt(F.mse_loss(modified_inputs[sample_mapping], batch_inputs)).item(),
                'targets/rmse/mean': torch.sqrt(F.mse_loss(modified_targets[sample_mapping], batch_targets)).item(),
                'inputs/mae/mean': F.l1_loss(modified_inputs[sample_mapping], batch_inputs).item(),
                'targets/mae/mean': F.l1_loss(modified_targets[sample_mapping], batch_targets).item(),
            }
            attack_metrics.update(mean_evaluation)

            # individual batch sample metrics
            for b_i, g_j in enumerate(sample_mapping):
                individual_evaluation = {
                    f'inputs/mse/{b_i}': F.mse_loss(modified_inputs[g_j], batch_inputs[b_i]).item(),
                    f'targets/mse/{b_i}': F.mse_loss(modified_targets[g_j], batch_targets[b_i]).item(),
                    f'inputs/rmse/{b_i}': torch.sqrt(F.mse_loss(modified_inputs[g_j], batch_inputs[b_i])).item(),
                    f'targets/rmse/{b_i}': torch.sqrt(F.mse_loss(modified_targets[g_j], batch_targets[b_i])).item(),
                    f'inputs/mae/{b_i}': F.l1_loss(modified_inputs[g_j], batch_inputs[b_i]).item(),
                    f'targets/mae/{b_i}': F.l1_loss(modified_targets[g_j], batch_targets[b_i]).item(),
                }
                attack_metrics.update(individual_evaluation)

            if attack_step % (num_attack_steps // 50) == 0 and verbose:
                print(f"Step {attack_step}/{num_attack_steps}, Loss: {dy_dx_loss}, evaluation: {mean_evaluation}")

            if attack_step % (num_attack_steps // 10) == 0:
                df, fig = self.plot_original_and_dummy_data(verbose, sample_mapping, modified_inputs, modified_targets, batch_inputs, batch_targets)
                self._log_dataframe(logger_object_dict, df, step=batch_number*num_attack_steps+attack_step)
                self._log_matplotlib_figure(logger_object_dict, fig, step=batch_number*num_attack_steps+attack_step)

            self._log_metrics(logger_object_dict, attack_metrics, step=batch_number*num_attack_steps+attack_step)
