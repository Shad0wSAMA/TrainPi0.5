"""Run LeRobot training with full, episode-local action chunks only."""


def configure_episode_safe_sampling(policy_config_class: type) -> None:
    # LeRobot's trainer reads this attribute when constructing EpisodeAwareSampler.
    # PI05Config does not declare it, so the upstream default is zero dropped frames.
    policy_config_class.drop_n_last_frames = property(lambda self: self.chunk_size - 1)


def main() -> None:
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    configure_episode_safe_sampling(PI05Config)

    from lerobot.scripts.lerobot_train import main as lerobot_main

    lerobot_main()


if __name__ == "__main__":
    main()
