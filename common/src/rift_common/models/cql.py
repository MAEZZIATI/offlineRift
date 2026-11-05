"""Optional d3rlpy registration for portable legacy CQL bundles."""

from dataclasses import dataclass

from d3rlpy.models.encoders import EncoderFactory, register_encoder_factory

from rift_common.models.encoders import TeamSetEncoder


@dataclass
class TeamSetEncoderFactory(EncoderFactory):
    grid_size: int = 4
    hidden_size: int = 128

    def create(self, observation_shape):
        return TeamSetEncoder(observation_shape[0], self.grid_size, self.hidden_size)

    @staticmethod
    def get_type() -> str:
        return "rift_team_set_v1"


register_encoder_factory(TeamSetEncoderFactory)
