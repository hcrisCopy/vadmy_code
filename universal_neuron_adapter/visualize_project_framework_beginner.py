from __future__ import annotations

import argparse
import html
from pathlib import Path


W, H = 1800, 1180


def text(x: int, y: int, value: str, size: int = 18, weight: int = 400,
         anchor: str = "middle", color: str = "#17324D") -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">'
        f'{html.escape(value)}</text>'
    )


def box(x: int, y: int, w: int, h: int, fill: str, title: str,
        lines: tuple[str, ...], edge: str = "#AFC0CC", title_size: int = 18,
        body_size: int = 15) -> str:
    items = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" '
        f'fill="{fill}" stroke="{edge}" stroke-width="2"/>',
        text(x + w // 2, y + 32, title, title_size, 700),
    ]
    start = y + h // 2 + 4 - (len(lines) - 1) * 11
    items.extend(text(x + w // 2, start + i * 23, line, body_size) for i, line in enumerate(lines))
    return "\n".join(items)


def band(y: int, h: int, title_value: str, fill: str = "#F7F9FB") -> str:
    return "\n".join([
        f'<rect x="20" y="{y}" width="1760" height="{h}" rx="20" fill="{fill}" stroke="#C3D0D9" stroke-width="2"/>',
        text(45, y + 38, title_value, 23, 700, "start"),
    ])


def arrow(x1: int, y1: int, x2: int, y2: int, label: str = "", dashed: bool = False,
          color: str = "#78909C") -> str:
    dash = ' stroke-dasharray="9 7"' if dashed else ""
    items = [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="3" marker-end="url(#arrow)"{dash}/>'
    ]
    if label:
        midx, midy = (x1 + x2) // 2, (y1 + y2) // 2 - 7
        tw = max(56, len(label) * 15)
        items.append(f'<rect x="{midx - tw // 2}" y="{midy - 17}" width="{tw}" height="23" rx="4" fill="white" opacity="0.92"/>')
        items.append(text(midx, midy, label, 14, color="#5B7083"))
    return "\n".join(items)


def render(output: Path) -> None:
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#78909C"/></marker></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Microsoft YaHei, Noto Sans CJK SC, SimHei, Arial, sans-serif">',
        text(900, 38, "VideoAnomalyDetection / vadmy_code 项目框架（初学者版）", 30, 700),
        text(900, 70, "核心目标：不改动原始异常检测器，用 CLIP 内部神经元证据校准它输出的异常分数", 19, color="#5B7083"),
        band(92, 185, "① 项目目录分别负责什么？"),
        box(55, 145, 300, 92, "#CDE7E7", "universal_neuron_adapter/", ("主方法（真正的核心）",), title_size=17),
        box(385, 145, 280, 92, "#D8E4F2", "baseline/", ("3 个被校准的基线复现",), title_size=17),
        box(695, 145, 280, 92, "#ECEFF1", "rely/", ("外部参考代码/论文实现",), title_size=17),
        box(1005, 145, 315, 92, "#FFF1C9", "innovation_evidence/", ("创新点验证小实验",), title_size=17),
        box(1350, 145, 375, 92, "#E8E6F5", "结果与实验记录", ("leaderboard / review / logs",), title_size=17),
        band(296, 375, "② 准备与训练：只使用官方训练集学习适配器", "#F7FAFC"),
        box(55, 405, 225, 135, "#D9EAF7", "输入数据", ("UCF-Crime / XD-Violence", "视频 + 视频级标签", "训练/测试 CSV")),
        box(330, 405, 245, 135, "#E8F0F7", "数据准备与防泄漏", ("data.py", "分训练/验证/测试", "SHA-256 + 3 项重叠审计")),
        box(625, 405, 245, 135, "#E8F0F7", "提取 CLIP 隐状态", ("extract_hidden_states.py", "每 16 帧一个片段", "输出 [T, 12, 768]")),
        arrow(280, 472, 330, 472, "清单"),
        arrow(575, 472, 625, 472, "视频片段"),
        box(925, 360, 235, 130, "#F8D4B4", "A. 主稀疏专家", ("Top-32 / 每层", "MIL 学异常坐标", "输出 expert score"), "#E7A45F"),
        box(1200, 360, 235, 130, "#E5DCF3", "B. 正态性专家", ("正常视频均值/方差", "保留有效偏离方向", "输出 normality score"), "#A991C7"),
        box(1475, 360, 235, 130, "#CFEDE2", "C. 上下文学生", ("当前 + 两个时间尺度", "线性学生补充上下文", "输出 context score"), "#5AA98C"),
        arrow(870, 450, 925, 425, "CLS 神经元"),
        arrow(870, 465, 1200, 425),
        arrow(870, 480, 1475, 425),
        box(925, 555, 325, 80, "#D8E4F2", "冻结基线得分缓存", ("cache_baseline.py → 1 条 baseline score",), title_size=17, body_size=14),
        box(1320, 555, 390, 80, "#E1EAF4", "分数修正头（每组合训练）", ("train_correction.py → correction checkpoint",), title_size=17, body_size=14),
        arrow(1250, 595, 1320, 595, "baseline + expert"),
        band(692, 360, "③ 正式推理与评估：每次只接收一个冻结基线", "#F8FBFA"),
        box(55, 805, 215, 135, "#D9EAF7", "测试输入", ("单个视频", "1 条冻结 baseline 分数", "CLS [T,12,768]")),
        box(315, 805, 230, 135, "#F3EEE8", "四路片段级信号", ("baseline score", "primary / context", "normality score")),
        box(590, 805, 245, 135, "#CDE7E7", "谱共识 + 保守融合", ("evaluate.py", "相关性主特征向量加权", "同意才增强，冲突少修改"), "#4D9C9C"),
        box(880, 805, 230, 135, "#F7D2CA", "时间恢复", ("中值/高斯平滑", "峰值边界扩张", "按训练持续长度定尺度"), "#D98273"),
        box(1155, 805, 215, 135, "#D4EFD9", "最终输出", ("片段异常分数", "×16 对齐为逐帧分数", "curves/*.npz"), "#5AA770"),
        box(1415, 805, 305, 135, "#FFF1C9", "评价与汇总", ("AUC / AP / detection mAP", "metrics.json / per_video.csv", "summary + 图表"), "#D0AE45"),
        arrow(270, 872, 315, 872), arrow(545, 872, 590, 872), arrow(835, 872, 880, 872),
        arrow(1110, 872, 1155, 872), arrow(1370, 872, 1415, 872),
        arrow(1040, 490, 430, 805, "训练好的 A", True, "#C6874A"),
        arrow(1315, 490, 455, 805, "训练好的 B", True, "#927BB0"),
        arrow(1590, 490, 480, 805, "训练好的 C", True, "#4C9279"),
        arrow(1515, 635, 710, 805, "修正头", True, "#6B7F93"),
        arrow(1085, 635, 160, 805, "单一基线", True, "#6B7F93"),
        text(900, 1085, "正式组合：2 个数据集 × 3 个冻结基线（LaGoVAD / DeSC / DSANet）= 6 组评估", 19, 700),
        text(900, 1118, "重要边界：测试标签只用于最后算指标；适配器不会读取第二个基线，也不更新原基线参数。", 17, color="#5B7083"),
        text(900, 1148, "读图方法：实线 = 数据流；虚线 = 训练后得到的模型/统计量被送入正式推理。", 16, color="#5B7083"),
        "</g></svg>",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "project_framework_beginner.svg").write_text("\n".join(svg), encoding="utf-8")
    print(f"[done] wrote framework SVG to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a beginner-friendly Chinese project framework.")
    parser.add_argument("--out-dir", default="docs/project_framework_beginner")
    args = parser.parse_args()
    output = Path(args.out_dir)
    if output.is_absolute():
        raise ValueError("--out-dir must be relative")
    render(output)


if __name__ == "__main__":
    main()
