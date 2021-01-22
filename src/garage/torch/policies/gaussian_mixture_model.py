import torch

from garage.torch.distributions.gmm import GMM
from garage.torch.policies.policy import Policy

EPS = 1e-6

class GMMPolicy(Policy, torch.nn.Module):
    def __init__(self, env_spec, K=2, hidden_layer_sizes=(256, 256), skills_num=0,
                 reg=1e-3, squash=True, reparameterize=False, qf=None,
                 name="GaussianMixtureModel"):
        self._skills_num = skills_num
        self._hidden_layers = hidden_layer_sizes
        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim + skills_num
        self._K = K
        self._is_deterministic = False
        self._fixed_h = None
        self._squash = squash
        self._qf = qf
        self._reg = reg

        assert not reparameterize
        self._reparameterize = reparameterize

        self.distribution = GMM(K=self._K,
                                hidden_layer_sizes=self._hidden_layers,
                                Dx=self._Da,
                                mlp_input_dim=self._Ds,
                                reg=self._reg)

        Policy.__init__(self, env_spec, name)
        torch.nn.Module.__init__(self)

    def get_action(self, observation):
        return self.get_actions(observation[None])[0]

    def get_action(self, observation, skill):
        return self.get_actions(torch.cat([observation[None], skill[None]], dim=1))[0]

    def forward(self, observations):
        log_p_x_t, reg_loss_t, x_t, log_ws_t, mus_t, log_sigs_t = self.distribution.get_p_params(
            observations)
        raw_actions = x_t.detach().cpu().numpy()
        actions = torch.tanh(raw_actions) if self._squash else raw_actions

        return actions, (log_p_x_t, reg_loss_t, x_t, log_ws_t, mus_t, log_sigs_t)

    def get_actions(self, observations):
        return self.forward(observations)

    def get_actions(self, observations, skills):
        return self.get_actions(torch.cat([observations, skills], dim=1))

    def _squash_correction(self, actions):
        if not self._squash:
            return 0
        else:
            return torch.sum(torch.log(1-torch.tanh(actions) ** 2 + EPS), dim=1)

    def parameters(self, recurse=False):
        return self.distribution.parameters()

    def reset(self):
        pass







