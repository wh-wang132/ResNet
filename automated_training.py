#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化训练脚本
用于系统地验证不同网络结构、批次大小和训练轮数对模型性能的影响
"""

import os
import sys
import json
import time
import logging
import subprocess
import csv
from datetime import datetime
from pathlib import Path


class AutomatedTrainer:
    """自动化训练器类"""

    def __init__(self):
        self.setup_logging()
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"experiments/{self.timestamp}"
        os.makedirs(self.experiment_dir, exist_ok=True)

        # 实验配置
        self.models = [
            "resnet6_2d",
            "resnet10_2d",
            "resnet14_2d",
            "resnet18_2d",
            "resnet34_2d",
            "resnet50_2d",
        ]

        self.batch_sizes = [8, 16, 32, 64, 128]
        self.epochs_list = [10, 20, 30, 50, 80, 120, 180]

        # ResNet50的最大批次大小限制
        self.resnet50_max_batch_size = 128

        self.logger.info("=" * 80)
        self.logger.info("自动化训练实验开始")
        self.logger.info(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)

    def setup_logging(self):
        """配置日志记录"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        log_file = f"{log_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    def get_max_batch_size(self, model):
        """获取模型的最大批次大小"""
        if model == "resnet50_2d":
            return self.resnet50_max_batch_size
        return max(self.batch_sizes)

    def run_training(self, model, batch_size, epochs):
        """运行单次训练"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(
            f"开始训练: 模型={model}, 批次大小={batch_size}, 训练轮数={epochs}"
        )
        self.logger.info(f"{'='*80}")

        start_time = time.time()
        result = {
            "model": model,
            "batch_size": batch_size,
            "epochs": epochs,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "pending",
            "test_acc": None,
            "best_val_acc": None,
            "training_time": None,
            "error_message": None,
        }

        try:
            # 构建命令（Linux命令）
            cmd = [
                "uv",
                "run",
                "python",
                "src/main.py",
                "--model",
                model,
                "--batch_size",
                str(batch_size),
                "--epochs",
                str(epochs),
                "--Train",
                "--Test",
                "--no-UMAP",
            ]

            self.logger.info(f"执行命令: {' '.join(cmd)}")

            # 运行训练并捕获输出（使用相对路径）
            # 脚本位于项目根目录，直接使用当前目录
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=".",
            )

            stdout, stderr = process.communicate()

            # 保存训练输出
            output_file = os.path.join(
                self.experiment_dir, f"output_{model}_bs{batch_size}_ep{epochs}.txt"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("STDOUT\n")
                f.write("=" * 80 + "\n")
                f.write(stdout)
                f.write("\n" + "=" * 80 + "\n")
                f.write("STDERR\n")
                f.write("=" * 80 + "\n")
                f.write(stderr)

            if process.returncode == 0:
                result["status"] = "success"
                self.logger.info("训练完成，正在提取性能指标...")

                # 从输出中提取指标（简化版本，实际项目中可能需要更复杂的解析）
                # 这里我们假设训练完成后会保存一些结果
                result["best_val_acc"] = self.extract_best_val_acc(stdout)
                result["test_acc"] = self.extract_test_acc(stdout)

                self.logger.info(f"最佳验证准确率: {result['best_val_acc']}")
                self.logger.info(f"测试准确率: {result['test_acc']}")
            else:
                result["status"] = "failed"
                result["error_message"] = stderr
                self.logger.error(f"训练失败: {stderr}")

        except Exception as e:
            result["status"] = "error"
            result["error_message"] = str(e)
            self.logger.error(f"训练过程出错: {str(e)}")

        end_time = time.time()
        result["training_time"] = round(end_time - start_time, 2)
        result["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.results.append(result)
        self.logger.info(f"训练耗时: {result['training_time']} 秒")
        self.logger.info(f"状态: {result['status']}")

        return result

    def extract_best_val_acc(self, stdout):
        """从输出中提取最佳验证准确率"""
        import re

        match = re.search(r"最佳验证准确率[:：]\s*([\d.]+)", stdout)
        if match:
            return float(match.group(1))
        return None

    def extract_test_acc(self, stdout):
        """从输出中提取测试准确率"""
        import re

        # 这里需要根据实际输出格式调整
        match = re.search(r"Test Acc[:：]\s*([\d.]+)", stdout)
        if match:
            return float(match.group(1))
        return None

    def save_results(self):
        """保存实验结果"""
        # 保存为JSON
        json_file = os.path.join(self.experiment_dir, "results.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # 保存为CSV
        csv_file = os.path.join(self.experiment_dir, "results.csv")
        if self.results:
            keys = self.results[0].keys()
            with open(csv_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.results)

        self.logger.info(f"\n实验结果已保存至: {self.experiment_dir}")
        return json_file, csv_file

    def generate_report(self):
        """生成实验报告"""
        report_file = os.path.join(self.experiment_dir, "experiment_report.md")

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# 自动化训练实验报告\n\n")
            f.write(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 实验配置\n\n")
            f.write(f"- 模型列表: {', '.join(self.models)}\n")
            f.write(f"- 批次大小: {', '.join(map(str, self.batch_sizes))}\n")
            f.write(f"- 训练轮数: {', '.join(map(str, self.epochs_list))}\n")
            f.write(f"- ResNet50最大批次大小: {self.resnet50_max_batch_size}\n\n")

            f.write("## 实验结果汇总\n\n")

            # 统计成功和失败数量
            success_count = sum(1 for r in self.results if r["status"] == "success")
            failed_count = sum(
                1 for r in self.results if r["status"] in ["failed", "error"]
            )

            f.write(f"- 总实验次数: {len(self.results)}\n")
            f.write(f"- 成功: {success_count}\n")
            f.write(f"- 失败: {failed_count}\n\n")

            f.write("## 详细结果\n\n")
            f.write(
                "| 模型 | 批次大小 | 训练轮数 | 状态 | 最佳验证准确率 | 测试准确率 | 训练时间(秒) |\n"
            )
            f.write(
                "|------|----------|----------|------|----------------|------------|--------------|\n"
            )

            for r in self.results:
                f.write(
                    f"| {r['model']} | {r['batch_size']} | {r['epochs']} | {r['status']} | "
                )
                f.write(
                    f"{r['best_val_acc'] or '-'} | {r['test_acc'] or '-'} | {r['training_time']} |\n"
                )

            f.write("\n## 性能分析\n\n")
            f.write("### 最佳性能模型\n\n")

            # 找出最佳模型
            successful_results = [
                r
                for r in self.results
                if r["status"] == "success" and r["best_val_acc"] is not None
            ]
            if successful_results:
                best_result = max(successful_results, key=lambda x: x["best_val_acc"])
                f.write(f"- **模型**: {best_result['model']}\n")
                f.write(f"- **批次大小**: {best_result['batch_size']}\n")
                f.write(f"- **训练轮数**: {best_result['epochs']}\n")
                f.write(f"- **最佳验证准确率**: {best_result['best_val_acc']}\n")
                f.write(f"- **训练时间**: {best_result['training_time']} 秒\n\n")

            f.write("## 结论\n\n")
            f.write("（请根据实际实验结果填写结论）\n")

        self.logger.info(f"实验报告已生成: {report_file}")
        return report_file

    def run_all_experiments(self):
        """运行所有实验"""
        total_experiments = 0

        # 计算总实验次数
        for model in self.models:
            max_bs = self.get_max_batch_size(model)
            valid_bs = [bs for bs in self.batch_sizes if bs <= max_bs]
            total_experiments += len(valid_bs) * len(self.epochs_list)

        self.logger.info(f"总实验次数: {total_experiments}")

        experiment_count = 0

        for model in self.models:
            max_bs = self.get_max_batch_size(model)
            valid_bs = [bs for bs in self.batch_sizes if bs <= max_bs]

            for batch_size in valid_bs:
                for epochs in self.epochs_list:
                    experiment_count += 1
                    self.logger.info(
                        f"\n实验进度: {experiment_count}/{total_experiments}"
                    )

                    self.run_training(model, batch_size, epochs)

                    # 实验间隔，释放资源
                    time.sleep(2)

        # 保存结果和生成报告
        self.save_results()
        self.generate_report()

        self.logger.info("\n" + "=" * 80)
        self.logger.info("所有实验完成！")
        self.logger.info("=" * 80)


def shutdown_system():
    """自动关机（Linux命令）"""
    import subprocess
    import logging

    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("实验完成，60秒后自动关机...")
    logger.info("如需取消关机，请按 Ctrl+C 并运行: sudo shutdown -c")
    logger.info("=" * 80)
    try:
        # Linux关机命令（免密sudo）
        subprocess.run(["sudo", "shutdown", "-h", "+1"], check=True)
    except Exception as e:
        logger.error(f"关机失败: {e}")


def main():
    """主函数"""
    trainer = AutomatedTrainer()
    trainer.run_all_experiments()
    # 实验完成后自动关机
    shutdown_system()


if __name__ == "__main__":
    main()
