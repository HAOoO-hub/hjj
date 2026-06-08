# graph_drawer.py：把自动机的状态图绘制到Tkinter画布上
from __future__ import annotations

import math
import tkinter as tk

# 全局缩放因子（用于放大缩小）
_graph_scale = 1.0


def set_graph_scale(scale: float) -> None:
    """Set the global graph scale factor."""
    global _graph_scale
    _graph_scale = max(0.5, min(3.0, scale))  # Limit scale between 0.5x and 3.0x


def get_graph_scale() -> float:
    """Get the current graph scale factor."""
    return _graph_scale


def build_layers(transitions: dict[str, dict[str, str]], start: str) -> list[list[str]]:
    """Build layers using BFS from start state."""
    seen = {start}
    layers = [[start]]
    frontier = [start]
    while frontier:
        nxt: list[str] = []
        for state in frontier:
            for target in transitions.get(state, {}).values():
                if target not in seen:
                    seen.add(target)
                    nxt.append(target)
        if nxt:
            layers.append(nxt)
        frontier = nxt
    remaining = [state for state in transitions if state not in seen]
    if remaining:
        layers.append(remaining)
    return layers


def calculate_grid_layout(
    transitions: dict[str, dict[str, str]],
    start: str,
    is_nfa: bool,
) -> dict[str, tuple[int, int]]:
    """Calculate positions using smart 2D layout with multi-column support."""
    layers = build_layers(transitions, start)
    
    # Spacing configuration
    if is_nfa:
        h_gap = 85   # Compact horizontal spacing for NFA
        v_gap = 60   # Compact vertical spacing for NFA
    else:
        h_gap = 120  # Moderate horizontal spacing for DFA
        v_gap = 85   # Moderate vertical spacing for DFA
    
    positions: dict[str, tuple[int, int]] = {}
    
    # Smart 2D layout: for large layers, split into multiple columns
    current_x = 100
    current_y_base = 80
    
    for layer_idx, layer in enumerate(layers):
        layer_size = len(layer)
        
        if layer_size <= 4:
            # Small layer: arrange vertically in one column
            for idx, state in enumerate(layer):
                y = current_y_base + idx * v_gap
                positions[state] = (current_x, y)
            current_x += h_gap
        else:
            # Large layer: split into 2-3 columns for compact layout
            cols = 3 if is_nfa else 2
            rows_per_col = (layer_size + cols - 1) // cols
            
            for idx, state in enumerate(layer):
                col = idx // rows_per_col
                row = idx % rows_per_col
                
                x = current_x + col * (h_gap // 2)
                y = current_y_base + row * v_gap
                positions[state] = (x, y)
            
            # Advance X by the width of this multi-column layer
            current_x += h_gap + (cols - 1) * (h_gap // 2)
    
    return positions


def draw_graph(
    canvas: tk.Canvas,
    start: str,
    accepts: set[str],
    transitions: dict[str, dict[str, str]],
    labels: dict[str, str] | None = None,
) -> None:
    """Draw automaton graph with optimized layout and beautiful color scheme."""
    canvas.delete("all")
    if not transitions:
        return
    
    scale = _graph_scale
    
    # Get all states including targets
    all_states = set(transitions.keys())
    for src in transitions:
        for dst in transitions[src].values():
            all_states.add(dst)
    all_states = list(all_states)
    
    # For NFA (single digit state IDs), show all states regardless of count
    # For DFA (letter state IDs), limit only if truly excessive
    is_nfa = all(any(c.isdigit() for c in state) for state in all_states[:5]) if all_states else False
    
    if not is_nfa and len(all_states) > 30:
        # Only limit large DFA graphs, not NFAs
        key_states = {start, *accepts}
        for state in list(key_states):
            for target in transitions.get(state, {}).values():
                key_states.add(target)
        transitions = {k: v for k, v in transitions.items() if k in key_states}
        all_states = list(set(transitions.keys()) | {t for row in transitions.values() for t in row.values()})
    
    layers = build_layers(transitions, start)
    
    # Optimized spacing: compact but no overlaps
    # For NFA: use smaller spacing (arrows will be shorter)
    # For DFA: use medium spacing
    is_nfa = all(any(c.isdigit() for c in state) for state in all_states[:5]) if all_states else False
    
    if is_nfa:
        # NFA: compact layout with smaller elements
        base_h_gap = 120  # Horizontal gap (smaller for NFA)
        base_v_gap = 80   # Vertical gap (smaller for NFA)
        node_radius = 22  # Smaller nodes for NFA
    else:
        # DFA: medium layout
        base_h_gap = 180  # Horizontal gap
        base_v_gap = 130  # Vertical gap
        node_radius = 28  # Normal nodes for DFA
    
    h_gap = int(base_h_gap * scale)
    v_gap = int(base_v_gap * scale)
    radius = int(node_radius * scale)
    
    # Use smart 2D grid layout
    positions = calculate_grid_layout(transitions, start, is_nfa)
    
    # Calculate canvas size based on actual positions
    if positions:
        min_x = min(x for x, y in positions.values())
        max_x = max(x for x, y in positions.values())
        min_y = min(y for x, y in positions.values())
        max_y = max(y for x, y in positions.values())
        
        # Dynamic canvas size with proper margins
        canvas_width = max_x - min_x + 300
        canvas_height = max(max_y - min_y + 300, 500 if is_nfa else 700)
    else:
        canvas_width = 800
        canvas_height = 600
    
    # Draw edges first (so they appear behind nodes)
    drawn_edges = set()
    
    # Group edges by (src, dst) to handle multi-edge situations better
    edge_groups: dict[tuple[str, str], list[tuple[str, int, int]]] = {}
    for src, row in transitions.items():
        for symbol, dst in row.items():
            if src not in positions or dst not in positions:
                continue
            edge_key = (src, dst)
            if edge_key not in edge_groups:
                edge_groups[edge_key] = []
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            edge_groups[edge_key].append((symbol, x1, y1, x2, y2))
    
    # Draw each edge group
    for (src, dst), edges in edge_groups.items():
        num_edges = len(edges)
        
        for idx, (symbol, x1, y1, x2, y2) in enumerate(edges):
            clean_symbol = symbol.split("#")[0] if "#" in symbol else symbol
            
            if src == dst:
                # Self-loop: draw above the node
                loop_size = int(20 * scale) if is_nfa else int(25 * scale)
                canvas.create_arc(x1 - loop_size, y1 - loop_size*1.5, 
                                x1 + loop_size, y1 - loop_size*0.5, 
                                start=20, extent=320, style="arc", outline="#6b7b8d", width=2)
                # Self-loop label
                font_size = int(8 * scale) if is_nfa else int(9 * scale)
                text_id = canvas.create_text(x1, y1 - loop_size*1.5 - 8, text=clean_symbol,
                                 font=("Microsoft YaHei", font_size), fill="#2d3748",
                                 justify="center")
                canvas.update_idletasks()
                bbox = canvas.bbox(text_id)
                if bbox:
                    padding = int(3 * scale) if is_nfa else int(4 * scale)
                    bg = canvas.create_rectangle(bbox[0]-padding, bbox[1]-int(2*scale),
                                                bbox[2]+padding, bbox[3]+int(2*scale),
                                                fill="#fefcbf", outline="", width=0)
                    canvas.tag_lower(bg, text_id)
            else:
                # Calculate arrow positions with offset from node center
                angle = math.atan2(y2 - y1, x2 - x1)
                sx = x1 + radius * math.cos(angle)
                sy = y1 + radius * math.sin(angle)
                ex = x2 - radius * math.cos(angle)
                ey = y2 - radius * math.sin(angle)
                
                # Calculate midpoint
                mid_x = (sx + ex) / 2
                mid_y = (sy + ey) / 2
                
                # For multiple edges between same nodes, use different curve amounts
                dx = x2 - x1
                dy = y2 - y1
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Determine if we need curves
                out_edges = len(transitions.get(src, {}))
                use_curve = out_edges > 1 or abs(dy) > v_gap * 0.3 or num_edges > 1
                
                if use_curve and distance > 0:
                    # Smaller curve for compact layout
                    base_curve = int(20 * scale) if is_nfa else int(30 * scale)
                    
                    # For multiple edges between same src-dst pair, add offset
                    if num_edges > 1:
                        edge_offset = (idx - (num_edges - 1) / 2) * int(15 * scale)
                        curve_amount = base_curve + edge_offset
                    else:
                        curve_amount = base_curve
                    
                    # Perpendicular offset for curve
                    offset_x = -dy / distance * curve_amount
                    offset_y = dx / distance * curve_amount
                    ctrl_x = mid_x + offset_x
                    ctrl_y = mid_y + offset_y
                    
                    # Curved arrow
                    canvas.create_line(sx, sy, ctrl_x, ctrl_y, ex, ey,
                                     arrow="last", fill="#6b7b8d", width=2.5, smooth=True)
                    label_x = ctrl_x
                    label_y = ctrl_y
                else:
                    # Straight line
                    canvas.create_line(sx, sy, ex, ey, arrow="last", fill="#6b7b8d", width=2.5)
                    label_x = mid_x
                    label_y = mid_y
                
                # Label positioning
                perp_angle = angle - math.pi/2
                label_offset = int(12 * scale) if is_nfa else int(18 * scale)
                label_x += label_offset * math.cos(perp_angle)
                label_y += label_offset * math.sin(perp_angle)
                
                # Draw label
                font_size = int(9 * scale) if is_nfa else int(10 * scale)
                text_id = canvas.create_text(label_x, label_y, text=clean_symbol,
                                 font=("Microsoft YaHei", font_size, "bold"),
                                 fill="#2d3748", justify="center")
                canvas.update_idletasks()
                bbox = canvas.bbox(text_id)
                if bbox:
                    padding = int(4 * scale) if is_nfa else int(5 * scale)
                    bg = canvas.create_rectangle(bbox[0]-padding, bbox[1]-int(2*scale),
                                                bbox[2]+padding, bbox[3]+int(2*scale),
                                                fill="#fefcbf", outline="", width=0)
                    canvas.tag_lower(bg, text_id)
    
    # Draw nodes with beautiful color scheme
    for state, (x, y) in positions.items():
        # Color scheme: start=green, accept=blue, regular=light gray
        if state == start:
            fill_color = "#c6f6d5"  # Light green
            outline_color = "#276749"  # Dark green
        elif state in accepts:
            fill_color = "#bee3f8"  # Light blue
            outline_color = "#2c5282"  # Dark blue
        else:
            fill_color = "#f7fafc"  # Very light gray
            outline_color = "#4a5568"  # Dark gray
        
        # Shadow for depth
        shadow_offset = int(2 * scale) if is_nfa else int(3 * scale)
        canvas.create_oval(x - radius + shadow_offset, y - radius + shadow_offset, 
                          x + radius + shadow_offset, y + radius + shadow_offset,
                          fill="#e2e8f0", outline="", width=0)
        
        # Main circle
        line_width = 2 if is_nfa else 2.5
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius, 
                          fill=fill_color, outline=outline_color, width=line_width)
        
        # Double circle for accept states
        if state in accepts:
            inner_radius = int(radius * 0.78)
            canvas.create_oval(x - inner_radius, y - inner_radius, 
                              x + inner_radius, y + inner_radius, 
                              fill=fill_color, outline=outline_color, width=line_width)
        
        # Start indicator
        if state == start:
            arrow_len = int(25 * scale) if is_nfa else int(35 * scale)
            canvas.create_line(x - arrow_len, y, x - radius, y, 
                             arrow="last", fill="#276749", width=line_width)
        
        # State label
        title = labels.get(state, state) if labels else state
        text_color = "#1a202c" if state in accepts else "#2d3748"
        font_size = int(10 * scale) if is_nfa else int(11 * scale)
        canvas.create_text(x, y, text=title, 
                          font=("Consolas", font_size, "bold"),
                          fill=text_color)
    
    # Set scroll region to fit content with proper padding
    if positions:
        min_x = min(x for x, y in positions.values())
        max_x = max(x for x, y in positions.values())
        min_y = min(y for x, y in positions.values())
        max_y = max(y for x, y in positions.values())
        padding = int(80 * _graph_scale)
        canvas.configure(scrollregion=(min_x - padding, min_y - padding, 
                                      max_x + padding, max_y + padding))
