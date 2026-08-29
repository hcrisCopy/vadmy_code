from __future__ import annotations

import argparse
import html
from pathlib import Path


W, H = 1900, 760


def label(x: int, y: int, value: str, size: int = 19, weight: int = 400,
          anchor: str = "middle", color: str = "#17324D") -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{html.escape(value)}</text>'
    )


def box(x: int, y: int, w: int, h: int, fill: str, title: str,
        lines: tuple[str, ...] = (), edge: str = "#9FB3C1",
        title_size: int = 21, body_size: int = 17) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" '
        f'fill="{fill}" stroke="{edge}" stroke-width="2.5"/>',
        label(x + w // 2, y + 40, title, title_size, 700),
    ]
    start = y + h // 2 + 17 - (len(lines) - 1) * 14
    parts.extend(label(x + w // 2, start + i * 29, item, body_size) for i, item in enumerate(lines))
    return "\n".join(parts)


def arrow(x1: int, y1: int, x2: int, y2: int, text_value: str,
          color: str = "#607D8B") -> str:
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    width = max(80, len(text_value) * 16)
    return "\n".join([
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        'stroke-width="3.5" marker-end="url(#arrow)"/>',
        f'<rect x="{mx - width // 2}" y="{my - 31}" width="{width}" height="27" rx="5" fill="white" opacity="0.94"/>',
        label(mx, my - 11, text_value, 15, color="#526B78"),
    ])


def render(output: Path) -> None:
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<defs><marker id="arrow" markerWidth="11" markerHeight="11" refX="10" refY="3.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,7 L10,3.5 z" fill="#607D8B"/></marker></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Microsoft YaHei, Noto Sans CJK SC, SimHei, Arial, sans-serif">',
        label(950, 44, "Universal CLS-Neuron Adapter", 31, 700),
        label(950, 76, "单一冻结检测器 + 三种互补神经元证据 → 逐帧异常分数", 18, color="#607585"),

        box(30, 300, 155, 120, "#DDECF7", "输入视频", ("Video",), "#79A8C7", 23, 18),
        box(225, 300, 170, 120, "#E7F0F6", "视频切片", ("每 16 帧", "一个 snippet"), "#91A8B8"),
        box(435, 300, 180, 120, "#E0EAF4", "冻结 CLIP", ("ViT-B/16", "12 层 CLS"), "#7F9DB5"),
        arrow(185, 360, 225, 360, "视频帧"),
        arrow(395, 360, 435, 360, "片段序列"),

        '<rect x="665" y="105" width="350" height="390" rx="24" fill="#FAFBFC" stroke="#AFC0CC" stroke-width="2.5"/>',
        label(840, 143, "三种互补神经元证据", 22, 700),
        box(700, 168, 280, 82, "#F8D9BD", "稀疏异常证据", ("Primary detector",), "#E6A35F", 19, 15),
        box(700, 275, 280, 82, "#D6EEE5", "多尺度上下文证据", ("Context detector",), "#63A98F", 19, 15),
        box(700, 382, 280, 82, "#E8DFF3", "方向性正常偏离", ("Normality detector",), "#9B86B6", 19, 15),
        arrow(615, 360, 665, 300, "CLS 状态 H(t)"),

        box(1065, 235, 205, 145, "#D2EBE9", "谱共识", ("正相关矩阵", "可靠性加权"), "#4C9A98", 23, 17),
        arrow(1015, 300, 1065, 307, "e₁(t), e₂(t), e₃(t)"),

        box(690, 565, 290, 115, "#DCE6F1", "冻结异常检测器", ("LaGoVAD / DeSC / DSANet",), "#7893AC", 21, 16),
        arrow(310, 420, 690, 622, "同一视频片段"),

        box(1320, 255, 205, 150, "#D5E6F3", "保守分数融合", ("一致时增强", "冲突时保留基线"), "#5F8CAE", 22, 17),
        arrow(1270, 307, 1320, 315, "共识证据 g(t)"),
        arrow(980, 622, 1365, 405, "基线分数 sᵦ(t)"),

        box(1570, 270, 170, 120, "#F7D8CF", "时序恢复", ("平滑", "边界扩张"), "#D77969", 22, 17),
        arrow(1525, 330, 1570, 330, "校准片段分数"),

        box(1780, 270, 95, 120, "#D8EFD9", "输出", ("逐帧", "异常分数"), "#5CA66B", 22, 17),
        arrow(1740, 330, 1780, 330, "ŝframe(t)"),

        label(950, 726, "图中每次只使用一个冻结检测器；三种神经元证据负责判断何时应该修改原始异常分数。", 17, color="#607585"),
        "</g></svg>",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "method_framework_simple.svg").write_text("\n".join(elements), encoding="utf-8")
    (output / "method_framework_simple_caption.txt").write_text(
        "方法框架。输入视频被划分为连续片段；冻结 CLIP 提供多层 CLS 隐状态，三种互补检测器分别提取稀疏异常、多尺度上下文和方向性正常偏离证据。谱共识对三路证据进行可靠性加权，并与单个冻结异常检测器的基线分数保守融合，最后经时序恢复输出逐帧异常分数。",
        encoding="utf-8",
    )
    print(f"[done] wrote simplified framework to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the simplified paper-style method framework.")
    parser.add_argument("--out-dir", default="docs/method_framework_simple")
    args = parser.parse_args()
    output = Path(args.out_dir)
    if output.is_absolute():
        raise ValueError("--out-dir must be relative")
    render(output)


if __name__ == "__main__":
    main()
