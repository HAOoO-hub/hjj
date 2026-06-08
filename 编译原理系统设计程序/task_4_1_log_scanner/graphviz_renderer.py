# graphviz_renderer.py: 使用graphviz渲染NFA/DFA到Tkinter界面
from __future__ import annotations

import os
import tempfile
from graphviz import Digraph
import tkinter as tk
from PIL import Image, ImageTk

from task_4_1_log_scanner.nfa import NFA, EPSILON
from task_4_1_log_scanner.dfa import DFA


def compress_symbol_label(symbols: set[str]) -> str:
    """Compress multiple symbols into readable ranges.
    
    Examples:
        {'0', '1', '2', ..., '9'} -> '0-9'
        {'A', 'B', ..., 'Z'} -> 'A-Z'
        {'0', '1', 'A'} -> '0,1,A'
    """
    if not symbols:
        return ""
    
    # Separate single characters from multi-character symbols
    single_chars = []
    multi_chars = []
    
    for sym in symbols:
        if len(sym) == 1:
            single_chars.append(sym)
        else:
            multi_chars.append(sym)
    
    if not single_chars:
        return ", ".join(sorted(multi_chars))
    
    # Sort single characters
    single_chars.sort()
    
    # Group consecutive characters into ranges
    ranges = []
    start = single_chars[0]
    end = single_chars[0]
    
    for ch in single_chars[1:]:
        if ord(ch) == ord(end) + 1:
            end = ch
        else:
            if start == end:
                ranges.append(start)
            else:
                ranges.append(f"{start}-{end}")
            start = ch
            end = ch
    
    # Add last range
    if start == end:
        ranges.append(start)
    else:
        ranges.append(f"{start}-{end}")
    
    # Combine ranges with multi-char symbols
    all_parts = ranges + sorted(multi_chars)
    return ", ".join(all_parts)


def nfa_to_dot(nfa: NFA, scale: float = 1.0) -> Digraph:
    """Convert NFA to graphviz Digraph with FIXED element sizes.
    
    All elements (nodes, fonts, edges) have ABSOLUTE fixed sizes
    regardless of graph complexity. This ensures consistent appearance
    across all graphs (small or large).
    
    The 'scale' parameter is IGNORED for element sizing - it only
    affects final display scaling via PIL/Pillow.
    """
    # FIXED absolute sizes - NOT scaled!
    node_size = 0.45  # Fixed node size in inches
    node_font = 10    # Fixed font size in points
    edge_font = 9     # Fixed edge font size in points
    penwidth_node = 1.6  # Fixed node border width
    penwidth_edge = 1.3  # Fixed edge width
    nodesep = 0.25    # Fixed horizontal spacing
    ranksep = 0.35    # Fixed vertical spacing
    
    dot = Digraph(
        engine='dot',
        graph_attr={
            'rankdir': 'LR',
            'bgcolor': '#fafbfc',
            'fontname': 'Microsoft YaHei',
            'fontsize': str(node_font),
            'labelloc': 't',
            'pad': '0.2',
            'nodesep': str(nodesep),
            'ranksep': str(ranksep),
        },
        node_attr={
            'shape': 'circle',
            'style': 'filled',
            'fillcolor': '#f7fafc',
            'color': '#4a5568',
            'fontname': 'Microsoft YaHei',
            'fontsize': str(node_font),
            'penwidth': str(penwidth_node),
            'width': str(node_size),
            'height': str(node_size),
            'fixedsize': 'true',
        },
        edge_attr={
            'fontname': 'Microsoft YaHei',
            'fontsize': str(edge_font),
            'penwidth': str(penwidth_edge),
            'color': '#718096',
            'minlen': '1',
        }
    )
    
    # Group transitions by (src, dst) to compress labels
    from collections import defaultdict
    edge_map: dict[tuple[str, str], set[str]] = defaultdict(set)
    
    for sid, state in nfa.states.items():
        for symbol, targets in state.transitions.items():
            for target in targets:
                edge_map[(str(sid), str(target))].add(symbol)
    
    # Add edges with compressed labels
    for (src, dst), symbols in edge_map.items():
        # Handle epsilon specially
        if EPSILON in symbols:
            label = 'ε'
        else:
            label = compress_symbol_label(symbols)
        
        dot.edge(src, dst, label=label)
    
    # Mark start state
    dot.node(str(nfa.start), fillcolor='#c6f6d5', color='#276749')
    
    # Mark accept state with double circle
    dot.node(str(nfa.accept), shape='doublecircle', fillcolor='#bee3f8', color='#2c5282')
    
    # Add invisible start arrow
    start_marker = f'start_{nfa.start}'
    dot.node(start_marker, shape='plaintext', label='', width='0', height='0')
    dot.edge(start_marker, str(nfa.start), penwidth='2')
    
    return dot


def dfa_to_dot(dfa: DFA, scale: float = 1.0, show_state_sets: bool = False) -> Digraph:
    """Convert DFA to graphviz Digraph with FIXED element sizes.
    
    All elements have ABSOLUTE fixed sizes to ensure consistency
    across all graphs (NFA/DFA/MinDFA).
    """
    # FIXED absolute sizes - NOT scaled! (same as NFA)
    node_size = 0.45  # Fixed node size in inches
    node_font = 10    # Fixed font size in points
    edge_font = 9     # Fixed edge font size in points
    penwidth_node = 1.6  # Fixed node border width
    penwidth_edge = 1.3  # Fixed edge width
    nodesep = 0.25    # Fixed horizontal spacing
    ranksep = 0.35    # Fixed vertical spacing
    
    dot = Digraph(
        engine='dot',
        graph_attr={
            'rankdir': 'LR',
            'bgcolor': '#fafbfc',
            'fontname': 'Microsoft YaHei',
            'fontsize': str(node_font),
            'labelloc': 't',
            'pad': '0.2',
            'nodesep': str(nodesep),
            'ranksep': str(ranksep),
        },
        node_attr={
            'shape': 'circle',
            'style': 'filled',
            'fillcolor': '#f7fafc',
            'color': '#4a5568',
            'fontname': 'Microsoft YaHei',
            'fontsize': str(node_font),
            'penwidth': str(penwidth_node),
            'width': str(node_size),
            'height': str(node_size),
            'fixedsize': 'true',
        },
        edge_attr={
            'fontname': 'Microsoft YaHei',
            'fontsize': str(edge_font),
            'penwidth': str(penwidth_edge),
            'color': '#718096',
            'minlen': '1',
        }
    )
    
    # Add transitions
    for state, trans in dfa.transitions.items():
        # Group by target state
        target_map: dict[str, set[str]] = {}
        for symbol, dst in trans.items():
            if dst not in target_map:
                target_map[dst] = set()
            target_map[dst].add(symbol)
        
        # Add edges with compressed labels
        for dst, symbols in target_map.items():
            label = compress_symbol_label(symbols)
            dot.edge(state, dst, label=label)
    
    # Mark start state
    if dfa.start in dfa.transitions:
        dot.node(dfa.start, fillcolor='#c6f6d5', color='#276749')
    
    # Mark accept states
    for accept in dfa.accepts:
        dot.node(accept, shape='doublecircle', fillcolor='#bee3f8', color='#2c5282')
    
    # Add invisible start arrow
    start_marker = f'start_{dfa.start}'
    dot.node(start_marker, shape='plaintext', label='', width='0', height='0')
    dot.edge(start_marker, dfa.start, penwidth='2')
    
    return dot


# Global cache for rendered PNGs to avoid re-generating on every zoom
_render_cache: dict[str, Image.Image] = {}
_cache_key_counter = 0

def _generate_cached_png(dot: Digraph) -> Image.Image:
    """Generate PNG from DOT, cache it to avoid re-generation.
    
    This is KEY for performance - only call Graphviz once!
    Subsequent zooms use cached image (instant!).
    """
    global _cache_key_counter
    
    # Create a unique key based on the DOT source
    dot_source = dot.source
    cache_key = hash(dot_source)
    
    # Check if already cached
    if cache_key in _render_cache:
        print(f"[Cache] ✓ HIT - using cached PNG (instant!)")
        return _render_cache[cache_key]
    
    print(f"[Cache] ✗ MISS - generating new PNG (slow, first time only)")
    print(f"[Cache] Cache size: {len(_render_cache)} items")
    # Generate new PNG (slow, but only once!)
    tmpdir = tempfile.mkdtemp()
    dot_path = os.path.join(tmpdir, f'automaton_{_cache_key_counter}')
    png_path = dot_path + '.png'
    
    # Render to PNG at HIGH quality (DPI=300 is good enough for screen display)
    dot.attr(dpi='300')  # High DPI - balanced quality vs speed
    dot.render(dot_path, format='png', cleanup=False)
    
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"PNG file not created: {png_path}")
    
    # Load and cache the image
    img = Image.open(png_path)
    _render_cache[cache_key] = img.copy()  # Keep a copy in memory
    _cache_key_counter += 1
    
    # Cleanup temp files
    try:
        os.remove(png_path)
        os.remove(dot_path)
    except:
        pass
    
    return _render_cache[cache_key]

def render_dot_to_tk(dot: Digraph, canvas: tk.Canvas, display_scale: float = 1.0) -> None:
    """Render graphviz Digraph to Tkinter canvas.
    
    Uses cached PNG for instant zooming - no Graphviz re-generation!
    Automatically adjusts display size based on graph complexity
    to ensure elements appear consistent across all graphs.
    """
    import time
    start_time = time.time()
    
    try:
        # Update canvas first to get proper size
        canvas.update_idletasks()
        canvas.update()
        
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            canvas_width = 800
            canvas_height = 600
        
        # Get cached PNG (instant!) or generate new one (first time only)
        t1 = time.time()
        img = _generate_cached_png(dot)
        t2 = time.time()
        print(f"[Cache] PNG retrieval: {(t2-t1)*1000:.1f}ms")
        
        original_width = img.width
        original_height = img.height
        
        # Calculate target display size based on display_scale
        # Simple and predictable: direct scaling, no complexity adjustment
        padding = 30
        max_width = canvas_width - padding
        max_height = canvas_height - padding
        
        # Base height: 60% of canvas at display_scale=1.0
        base_height = int(max_height * 0.60)
        base_height = max(base_height, 200)  # Minimum 200px for base
        
        # Apply display_scale directly
        target_height = int(base_height * display_scale)
        # Very low minimum (50px) to allow full zoom range
        target_height = max(target_height, 50)
        target_height = min(target_height, max_height)  # Maximum canvas height
        
        # Scale based on height
        img_ratio = original_width / original_height
        scale_factor = target_height / original_height
        new_height = target_height
        new_width = int(original_width * scale_factor)
        
        # Resize the image with high-quality LANCZOS resampling
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        print(f"[Graphviz] {original_width}x{original_height} → {new_width}x{new_height} "
              f"(display_scale: {display_scale:.3f})")
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(img)
        
        # Clear canvas and display
        canvas.delete("all")
        
        # Display image at center of scrollable area
        canvas.create_image(new_width // 2, new_height // 2, image=photo, anchor='center')
        
        # Set scrollregion to match image size (allow scrolling)
        canvas.config(scrollregion=(0, 0, new_width, new_height))
        
        # CRITICAL: Center the image in the viewport
        # Calculate offset to center the image
        if new_width > canvas_width or new_height > canvas_height:
            # Center horizontally
            offset_x = max(0, (new_width - canvas_width) / 2)
            canvas.xview_moveto(offset_x / new_width)
            
            # Center vertically  
            offset_y = max(0, (new_height - canvas_height) / 2)
            canvas.yview_moveto(offset_y / new_height)
        
        # CRITICAL: Keep MULTIPLE references to prevent garbage collection
        canvas._graph_image = photo
        canvas.image = photo
        if not hasattr(canvas, '_image_references'):
            canvas._image_references = []
        canvas._image_references.append(photo)
        
        # Store image dimensions for reference
        canvas._image_width = new_width
        canvas._image_height = new_height
        
        canvas.update_idletasks()
        canvas.update()
        
        elapsed = (time.time() - start_time) * 1000
        print(f"[Performance] Total render time: {elapsed:.1f}ms")
        
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stdout)
        
        # Show error on canvas
        canvas.delete("all")
        error_text = f"Graphviz渲染失败\n{str(e)[:80]}"
        canvas.create_text(
            canvas.winfo_width() // 2,
            canvas.winfo_height() // 2,
            text=error_text,
            font=("Microsoft YaHei", 11),
            fill="red",
            justify="center"
        )
