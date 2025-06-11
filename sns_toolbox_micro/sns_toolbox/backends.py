"""
Simulation backends for synthetic nervous system networks. Each of these are python-based, and are constructed using a
Network. They can then be run for a step, with the inputs being a vector of neural states and applied currents and the
output being the next step of neural states.
"""

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""
from ulab import numpy as np

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BACKENDS
"""

class Backend:

    def __init__(self, params: dict) -> None:
        self.set_params(params)

    def set_params(self, params: dict) -> None:
        self.dt = params['dt']
        self.name = params['name']
        self.spiking = params['spiking']
        self.delay = params['delay']
        self.electrical = params['elec']
        self.electrical_rectified = params['rect']
        self.gated = params['gated']
        self.num_channels = params['numChannels']
        self.V = params['v']
        self.V_last = params['vLast']
        self.V_0 = params['v0']
        self.V_rest = params['vRest']
        self.c_m = params['cM']
        self.g_m = params['gM']
        self.i_b = params['iB']
        self.g_max_non = params['gMaxNon']
        self.del_e = params['delE']
        self.e_lo = params['eLo']
        self.e_hi = params['eHi']
        self.time_factor_membrane = params['timeFactorMembrane']
        self.input_connectivity = params['inputConn']
        self.output_voltage_connectivity = params['outConnVolt']
        self.num_populations = params['numPop']
        self.num_neurons = params['numNeurons']
        self.num_connections = params['numConn']
        self.num_inputs = params['numInputs']
        self.num_outputs = params['numOutputs']
        # self.R = params['r']
        if self.spiking:
            self.spikes = params['spikes']
            self.theta_0 = params['theta0']
            self.theta = params['theta']
            self.theta_last = params['thetaLast']
            self.m = params['m']
            self.tau_theta = params['tauTheta']
            self.g_max_spike = params['gMaxSpike']
            self.g_spike = params['gSpike']
            self.tau_syn = params['tauSyn']
            self.time_factor_threshold = params['timeFactorThreshold']
            self.time_factor_synapse = params['timeFactorSynapse']
            self.output_spike_connectivity = params['outConnSpike']
            self.theta_leak = params['thetaLeak']
            self.theta_increment = params['thetaIncrement']
            self.theta_floor = params['thetaFloor']
            self.V_reset = params['vReset']
            self.g_increment = params['gIncrement']
        if self.delay:
            self.spike_delays = params['spikeDelays']
            self.spike_rows = params['spikeRows']
            self.spike_cols = params['spikeCols']
            self.buffer_steps = params['bufferSteps']
            self.buffer_nrns = params['bufferNrns']
            self.delayed_spikes = params['delayedSpikes']
            self.spike_buffer = params['spikeBuffer']
        if self.electrical:
            self.g_electrical = params['gElectrical']
        if self.electrical_rectified:
            self.g_rectified = params['gRectified']
        if self.gated:
            self.g_ion = params['gIon']
            self.e_ion = params['eIon']
            self.pow_a = params['powA']
            self.slope_a = params['slopeA']
            self.k_a = params['kA']
            self.e_a = params['eA']
            self.pow_b = params['powB']
            self.slope_b = params['slopeB']
            self.k_b = params['kB']
            self.e_b = params['eB']
            self.tau_max_b = params['tauMaxB']
            self.pow_c = params['powC']
            self.slope_c = params['slopeC']
            self.k_c = params['kC']
            self.e_c = params['eC']
            self.tau_max_c = params['tauMaxC']
            self.b_gate = params['bGate']
            self.b_gate_last = params['bGateLast']
            self.b_gate_0 = params['bGate0']
            self.c_gate = params['cGate']
            self.c_gate_last = params['cGateLast']
            self.c_gate_0 = params['cGate0']

    def __call__(self, x=None):
        return self.forward(x)

class SNS_Numpy(Backend):
    def __init__(self, params: dict) -> None:
        super().__init__(params)

    def forward(self, x=None):
        self.V_last = np.array(self.V)
        if x is None:
            i_app = 0
        else:
            i_app = np.dot(self.input_connectivity, x)  # Apply external current sources to their destinations
        g_syn = np.maximum(0, np.minimum(self.g_max_non * ((self.V_last - self.e_lo) / (self.e_hi - self.e_lo)), self.g_max_non))
        if self.spiking:
            self.theta_last = np.array(self.theta)
            self.g_spike = self.g_spike * (1 - self.time_factor_synapse)
            g_syn += self.g_spike

        i_syn = np.sum(g_syn * self.del_e, axis=1) - self.V_last * np.sum(g_syn, axis=1)
        if self.electrical:
            i_syn += (np.sum(self.g_electrical * self.V_last, axis=1) - self.V_last * np.sum(self.g_electrical, axis=1))
        if self.electrical_rectified:
            # create mask
            mask = np.subtract.outer(self.V_last, self.V_last).transpose() > 0
            masked_g = mask * self.g_rectified
            diag_masked = masked_g + masked_g.transpose() - np.diag(masked_g.diagonal())
            i_syn += np.sum(diag_masked * self.V_last, axis=1) - self.V_last * np.sum(diag_masked, axis=1)
        if self.gated:
            a_inf = 1 / (1 + self.k_a * np.exp(self.slope_a * (self.e_a - self.V_last)))
            b_inf = 1 / (1 + self.k_b * np.exp(self.slope_b * (self.e_b - self.V_last)))
            c_inf = 1 / (1 + self.k_c * np.exp(self.slope_c * (self.e_c - self.V_last)))

            tau_b = self.tau_max_b * b_inf * np.sqrt(self.k_b * np.exp(self.slope_b * (self.e_b - self.V_last)))
            tau_c = self.tau_max_c * c_inf * np.sqrt(self.k_c * np.exp(self.slope_c * (self.e_c - self.V_last)))

            self.b_gate_last = np.array(self.b_gate)
            self.c_gate_last = np.array(self.c_gate)

            self.b_gate = self.b_gate_last + self.dt * ((b_inf - self.b_gate_last) / tau_b)
            self.c_gate = self.c_gate_last + self.dt * ((c_inf - self.c_gate_last) / tau_c)

            i_ion = self.g_ion * (a_inf ** self.pow_a) * (self.b_gate ** self.pow_b) * (self.c_gate ** self.pow_c) * (
                        self.e_ion - self.V_last)
            i_gated = np.sum(i_ion, axis=0)

            self.V = self.V_last + self.time_factor_membrane * (
                        -self.g_m * (self.V_last - self.V_rest) + self.i_b + i_syn + i_app + i_gated)  # Update membrane potential
        else:
            self.V = self.V_last + self.time_factor_membrane * (
                        -self.g_m * (self.V_last - self.V_rest) + self.i_b + i_syn + i_app)  # Update membrane potential
        if self.spiking:
            self.theta = self.theta_last + self.time_factor_threshold * (self.theta_leak*(self.theta_0-self.theta_last) + self.m * (self.V_last - self.V_rest))  # Update the firing thresholds
            self.spikes = np.sign(np.minimum(0, self.theta - self.V))  # Compute which neurons have spiked

            # New stuff with delay
            if self.delay:
                self.spike_buffer = np.roll(self.spike_buffer, 1, axis=0)  # Shift buffer entries down
                self.spike_buffer[0, :] = self.spikes  # Replace row 0 with the current spike data
                # Update a matrix with all of the appropriately delayed spike values
                self.delayed_spikes[self.spike_rows, self.spike_cols] = self.spike_buffer[
                    self.buffer_steps, self.buffer_nrns]

                self.g_spike += np.minimum((-self.delayed_spikes*self.g_increment), (-self.delayed_spikes)*(self.g_max_spike-self.g_spike))  # Update the conductance of connections which spiked
            else:
                self.g_spike += np.minimum((-self.spikes*self.g_increment), (-self.spikes)*(self.g_max_spike-self.g_spike))  # Update the conductance of connections which spiked
            self.V = ((self.V-self.V_reset) * (self.spikes + 1))+self.V_reset  # Reset the membrane voltages of neurons which spiked
            self.theta = np.maximum(self.theta_increment, self.theta_floor-self.theta)*(-self.spikes) + self.theta
        self.outputs = np.dot(self.output_voltage_connectivity, self.V)
        if self.spiking:
            self.outputs += np.dot(self.output_spike_connectivity, -self.spikes)

        return self.outputs

    def reset(self):
        self.V = np.array(self.V_0)
        self.V_last = np.array(self.V_0)
        if self.spiking:
            self.theta = np.array(self.theta_0)
            self.theta_last = np.array(self.theta_0)
        if self.gated:
            self.b_gate = np.array(self.b_gate_0)
            self.b_gate_last = np.array(self.b_gate_0)
            self.c_gate = np.array(self.c_gate_0)
            self.c_gate_last = np.array(self.c_gate_0)