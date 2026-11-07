# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 OpenAI
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/openai/mle-bench/blob/main/LICENSE

# 导入命令行参数解析模块
import argparse
# 导入 os，用于删除原始压缩包文件
import os
# 导入 Path，用于路径拼接与解析
from pathlib import Path

# 导入 mlebench 的数据集下载与准备函数
from mlebench.data import (
    download_and_prepare_dataset,
)
# 导入 mlebench 的比赛注册表
from mlebench.registry import registry
# 导入 mlebench 的日志工具
from mlebench.utils import (
    get_logger,
)
# 导入本任务的通用数据工具（读取划分、解压、打包）
import dojo.tasks.mlebench.utils.data as data_utils

# 创建模块级日志器
logger = get_logger(__name__)


def main():
    # 命令行入口：为 Kaggle 竞赛准备数据集
    parser_prepare = argparse.ArgumentParser(description="Runs agents on Kaggle competitions.")
    # 参数 -c/--competition-id：要准备的单个比赛 ID
    parser_prepare.add_argument(
        "-c",
        "--competition-id",
        help=f"ID of the competition to prepare. Valid options: {registry.list_competition_ids()}",
        type=str,
        required=False,
    )
    # 参数 -s/--split：按 splits 目录下的划分文件批量准备比赛
    parser_prepare.add_argument(
        "-s",
        "--split",
        help="Prepare a list of competitions specified line by line in the mlebench/splits directory.",
        type=str,
        required=False,
    )
    # 开关 --keep-raw：准备完成后保留原始竞赛文件
    parser_prepare.add_argument(
        "--keep-raw",
        help="Keep the raw competition files after the competition has been prepared.",
        action="store_true",
        required=False,
        default=False,
    )
    # 参数 --data-dir：数据存放目录，默认为共享缓存路径
    parser_prepare.add_argument(
        "--data-dir",
        help="Path to the directory where the data will be stored.",
        required=False,
        default="shared/cache/dojo/tasks/mlebench",
    )
    # 开关 --keep-zip：准备完成后保留原始 zip 压缩包
    parser_prepare.add_argument(
        "--keep-zip",
        help="Keep the zipped raw competition files after the competition has been prepared.",
        action="store_true",
        required=False,
        default=False,
    )
    # 开关 --overwrite-checksums：开发者用，覆盖比赛的校验和文件
    parser_prepare.add_argument(
        "--overwrite-checksums",
        help="[For Developers] Overwrite the checksums file for the competition.",
        action="store_true",
        required=False,
        default=False,
    )
    # 开关 --overwrite-leaderboard：开发者用，覆盖比赛排行榜文件
    parser_prepare.add_argument(
        "--overwrite-leaderboard",
        help="[For Developers] Overwrite the leaderboard file for the competition.",
        action="store_true",
        required=False,
        default=False,
    )
    # 开关 --skip-verification：开发者用，跳过校验和验证
    parser_prepare.add_argument(
        "--skip-verification",
        help="[For Developers] Skip the verification of the checksums.",
        action="store_true",
        required=False,
        default=False,
    )
    # 解析命令行参数
    args = parser_prepare.parse_args()

    # 让注册表指向用户指定的数据目录，得到新的注册表实例
    new_registry = registry.set_data_dir(Path(args.data_dir))

    # 依据参数确定待处理的比赛列表
    if args.split:
        # 指定了划分：读取划分文件中的比赛 ID 列表
        competition_ids = data_utils.get_competition_ids_in_split(args.split)
        # 逐个从注册表取出对应的比赛对象
        competitions = [new_registry.get_competition(competition_id) for competition_id in competition_ids]
    else:
        # 未指定划分时必须给出比赛 ID，否则报错退出
        if not args.competition_id:
            parser_prepare.error("One of --competition-id or --split must be specified.")
        # 只处理单个比赛
        competitions = [new_registry.get_competition(args.competition_id)]

    # 逐个比赛执行数据准备流程
    for competition in competitions:
        # from mlebench.utils import (
        #     authenticate_kaggle_api,
        # )
        # api = authenticate_kaggle_api()

        # # only import when necessary; otherwise kaggle asks for API key on import
        # from kaggle.rest import ApiException

        # try:
        #     api.competition_download_files(
        #         competition=competition.id,
        #         path=".",
        #         quiet=True,
        #         force=True,
        #     )
        # except ApiException as e:
        #     if _need_to_accept_rules(str(e)):
        #         logger.warning("You must accept the competition rules before downloading the dataset.")
        #         _prompt_user_to_accept_rules(competition.id)
        #         # download_dataset(competition_id, download_dir, quiet, force)
        #     else:
        #         raise e
        # except Exception as e:
        #     print(e)

        # Support for custom prepare_fn logic
        # object.__setattr__(competition, "prepare_fn", import_fn("data.mlebench.aptos2019-blindness-detection.prepare:prepare"))
        # 下载并准备该比赛的数据集（含校验和与排行榜处理选项）
        download_and_prepare_dataset(
            competition=competition,
            keep_raw=args.keep_raw,
            overwrite_checksums=args.overwrite_checksums,
            overwrite_leaderboard=args.overwrite_leaderboard,
            skip_verification=args.skip_verification,
        )

        # 拼出该比赛原始 zip 压缩包的路径
        raw_zip_path = Path(args.data_dir).resolve() / competition.id / f"{competition.id}.zip"

        # 未要求保留 zip 且文件存在时，删除原始压缩包以节省空间
        if not args.keep_zip and raw_zip_path.exists():
            os.remove(raw_zip_path)

        # Our Agents expect the data to be already extracted
        # 定位准备后的公开数据目录（智能体训练时直接使用）
        path_to_public_folder = Path(args.data_dir).resolve() / competition.id / "prepared" / "public"
        # 递归解压该目录下所有压缩文件；未要求保留 zip 时顺带删除压缩包
        data_utils.extract_all_from_path(
            path=path_to_public_folder,
            delete_compressed=not args.keep_zip,
        )

        # tarball the public folder
        # 将公开文件夹整体打包为 public.tar，便于一次性挂载/分发
        tarball_path = path_to_public_folder.parent / "public.tar"
        data_utils.tar_directory(root_dir=path_to_public_folder, output_file=tarball_path)


# 脚本直接运行时执行数据准备主流程
if __name__ == "__main__":
    main()
