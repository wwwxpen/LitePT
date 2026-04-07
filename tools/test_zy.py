from engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from engines.test_zy import TESTERS
from engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    tester.test()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    try:
        launch(
            main_worker,
            num_gpus_per_machine=args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            cfg=(cfg,),
        )
    except Exception:
        print(f"test_zy.py运行出错")
        raise

    return 0


if __name__ == "__main__":
    main()
