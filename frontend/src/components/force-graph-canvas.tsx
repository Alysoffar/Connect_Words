"use client";

import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import {
  useEffect,
  useRef,
  type PointerEvent as ReactPointerEvent,
} from "react";

export type NodeKind = "anchor-a" | "anchor-b" | "word";

export interface GraphNode {
  id: string;
  kind: NodeKind;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  nearMiss?: boolean;
  expiresAt?: number;
}

interface ForceNode extends SimulationNodeDatum, GraphNode {}

interface ForceLink extends SimulationLinkDatum<ForceNode> {
  weight: number;
  nearMiss?: boolean;
  expiresAt?: number;
}

interface ForceGraphCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  winningPath: string[] | null;
  selectedNode: string | null;
  onSelectNode: (word: string) => void;
  height?: number;
}

function edgeKey(first: string, second: string): string {
  return first < second ? `${first}\u0000${second}` : `${second}\u0000${first}`;
}

function nodePosition(
  node: ForceNode,
  width: number,
  height: number,
): { x: number; y: number } {
  return {
    x: node.x ?? width / 2,
    y: node.y ?? height / 2,
  };
}

function linkNodeId(node: string | number | ForceNode): string {
  return typeof node === "object" ? node.id : String(node);
}

function linkNodePosition(
  node: string | number | ForceNode,
  points: Map<string, { x: number; y: number }>,
  width: number,
  height: number,
): { x: number; y: number } | undefined {
  return typeof node === "object"
    ? nodePosition(node, width, height)
    : points.get(String(node));
}

export function ForceGraphCanvas({
  nodes,
  edges,
  winningPath,
  selectedNode,
  onSelectNode,
  height = 440,
}: ForceGraphCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simulationRef = useRef<Simulation<ForceNode, ForceLink> | null>(null);
  const positionsRef = useRef(new Map<string, { x: number; y: number }>());
  const selectRef = useRef(onSelectNode);
  const winningPathRef = useRef(winningPath);
  const selectedRef = useRef(selectedNode);
  const pointerRef = useRef<{
    node: ForceNode;
    startX: number;
    startY: number;
    moved: boolean;
  } | null>(null);

  useEffect(() => {
    selectRef.current = onSelectNode;
    winningPathRef.current = winningPath;
    selectedRef.current = selectedNode;
  }, [onSelectNode, selectedNode, winningPath]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    const retainedPositions = positionsRef.current;
    const simulationNodes: ForceNode[] = nodes.map((node) => {
      const position = retainedPositions.get(node.id);
      return {
        ...node,
        ...(position ? { x: position.x, y: position.y } : {}),
      };
    });
    const simulationLinks: ForceLink[] = edges.map((edge) => ({
      ...edge,
      source: edge.source,
      target: edge.target,
    }));

    let width = 0;
    let pixelRatio = 1;
    let frame = 0;

    const resizeCanvas = () => {
      const bounds = canvas.getBoundingClientRect();
      width = Math.max(bounds.width, 1);
      pixelRatio = Math.max(window.devicePixelRatio || 1, 1);
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      simulation.force("center", forceCenter(width / 2, height / 2));
      simulation.alpha(0.35).restart();
    };

    const positions = () => {
      const result = new Map<string, { x: number; y: number }>();
      for (const node of simulationNodes) {
        result.set(node.id, nodePosition(node, width, height));
      }
      return result;
    };

    const draw = (timestamp: number) => {
      if (!width) {
        return;
      }
      context.clearRect(0, 0, width, height);
      const points = positions();
      const winningEdges = new Set<string>();
      const path = winningPathRef.current;
      if (path) {
        for (let index = 0; index < path.length - 1; index += 1) {
          winningEdges.add(edgeKey(path[index], path[index + 1]));
        }
      }

      for (const link of simulationLinks) {
        const source = linkNodePosition(link.source, points, width, height);
        const target = linkNodePosition(link.target, points, width, height);
        if (!source || !target) {
          continue;
        }
        const isNearMiss = link.nearMiss === true;
        const remaining = link.expiresAt ? link.expiresAt - Date.now() : 0;
        if (isNearMiss && remaining <= 0) {
          continue;
        }
        const pathEdge = !isNearMiss && winningEdges.has(
          edgeKey(
            linkNodeId(link.source),
            linkNodeId(link.target),
          ),
        );
        context.save();
        context.globalAlpha = isNearMiss
          ? Math.max(0.08, Math.min(0.58, remaining / 2000))
          : 1;
        context.beginPath();
        context.moveTo(source.x, source.y);
        context.lineTo(target.x, target.y);
        context.lineWidth = isNearMiss ? 1.5 : 1.25 + Math.max(0, link.weight) * 3;
        context.strokeStyle = isNearMiss ? "#687f87" : "#79918a";
        context.setLineDash(isNearMiss ? [2, 6] : []);
        context.stroke();
        context.setLineDash([]);

        if (pathEdge) {
          const gradient = context.createLinearGradient(
            source.x,
            source.y,
            target.x,
            target.y,
          );
          gradient.addColorStop(0, "#009e73");
          gradient.addColorStop(1, "#e69f00");
          context.beginPath();
          context.moveTo(source.x, source.y);
          context.lineTo(target.x, target.y);
          context.strokeStyle = gradient;
          context.lineWidth = 5;
          context.lineCap = "round";
          context.shadowColor = "#e69f00";
          context.shadowBlur = 9;
          context.setLineDash([11, 6]);
          context.lineDashOffset = -timestamp / 90;
          context.stroke();
          context.setLineDash([]);
          context.shadowBlur = 0;
        }

        const label = `${Math.round(link.weight * 100)}%`;
        const labelX = (source.x + target.x) / 2;
        const labelY = (source.y + target.y) / 2 - 7;
        context.font = "600 11px system-ui, sans-serif";
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.lineWidth = 4;
        context.strokeStyle = "#f5f7f2";
        context.strokeText(label, labelX, labelY);
        context.fillStyle = isNearMiss ? "#455d65" : "#304b44";
        context.fillText(label, labelX, labelY);
        context.restore();
      }

      for (const node of simulationNodes) {
        const { x, y } = nodePosition(node, width, height);
        const isAnchorA = node.kind === "anchor-a";
        const isAnchorB = node.kind === "anchor-b";
        const isSelected = selectedRef.current === node.id;
        const isWinning = path?.includes(node.id) ?? false;
        const radius = isAnchorA || isAnchorB ? 18 : 14;
        context.save();
        context.beginPath();
        if (isAnchorB) {
          context.moveTo(x, y - radius);
          context.lineTo(x + radius, y);
          context.lineTo(x, y + radius);
          context.lineTo(x - radius, y);
          context.closePath();
        } else {
          context.arc(x, y, radius, 0, Math.PI * 2);
        }
        context.fillStyle = isAnchorA
          ? "#0072b2"
          : isAnchorB
            ? "#d55e00"
            : isWinning
              ? "#d7f0e5"
              : "#ffffff";
        context.fill();
        context.lineWidth = isSelected ? 4 : isAnchorA || isAnchorB ? 3 : 2;
        context.strokeStyle = isSelected
          ? "#e69f00"
          : isAnchorA || isAnchorB
            ? "#ffffff"
            : isWinning
              ? "#009e73"
              : "#789089";
        context.stroke();
        context.font = `${isAnchorA || isAnchorB ? "700" : "600"} 12px system-ui, sans-serif`;
        context.textAlign = "center";
        context.textBaseline = "top";
        context.lineWidth = 4;
        context.strokeStyle = "#f5f7f2";
        context.strokeText(node.id, x, y + radius + 5);
        context.fillStyle = "#173c36";
        context.fillText(node.id, x, y + radius + 5);
        context.restore();
      }
      retainedPositions.clear();
      for (const [id, point] of points) {
        retainedPositions.set(id, point);
      }
      frame = window.requestAnimationFrame(draw);
    };

    const linkForce = forceLink<ForceNode, ForceLink>(simulationLinks)
      .id((node) => node.id)
      .distance((link) => 76 + (1 - Math.max(0, link.weight)) * 90)
      .strength((link) => 0.25 + Math.max(0, link.weight) * 0.4);
    const simulation = forceSimulation<ForceNode>(simulationNodes)
      .force("link", linkForce)
      .force("charge", forceManyBody<ForceNode>().strength(-250))
      .force("collide", forceCollide<ForceNode>(34))
      .force("center", forceCenter(width / 2, height / 2));
    simulationRef.current = simulation;

    const drawFrame = () => {
      if (frame) {
        window.cancelAnimationFrame(frame);
      }
      frame = window.requestAnimationFrame(draw);
    };
    simulation.on("tick", drawFrame);
    resizeCanvas();
    drawFrame();

    const observer = new ResizeObserver(resizeCanvas);
    observer.observe(canvas);

    return () => {
      observer.disconnect();
      simulation.stop();
      simulationRef.current = null;
      window.cancelAnimationFrame(frame);
    };
  }, [edges, height, nodes]);

  const pointerPosition = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return {
      x: event.clientX - bounds.left,
      y: event.clientY - bounds.top,
    };
  };

  const nearestNode = (
    event: ReactPointerEvent<HTMLCanvasElement>,
  ): ForceNode | undefined => {
    const point = pointerPosition(event);
    return simulationRef.current
      ?.nodes()
      .filter(
        (node) =>
          Math.hypot((node.x ?? 0) - point.x, (node.y ?? 0) - point.y) <
          (node.kind === "word" ? 20 : 25),
      )
      .sort(
        (first, second) =>
          Math.hypot((first.x ?? 0) - point.x, (first.y ?? 0) - point.y) -
          Math.hypot((second.x ?? 0) - point.x, (second.y ?? 0) - point.y),
      )[0];
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const node = nearestNode(event);
    if (!node) {
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    pointerRef.current = {
      node,
      startX: event.clientX,
      startY: event.clientY,
      moved: false,
    };
    const point = pointerPosition(event);
    node.fx = point.x;
    node.fy = point.y;
    simulationRef.current?.alphaTarget(0.25).restart();
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const drag = pointerRef.current;
    if (!drag) {
      return;
    }
    const point = pointerPosition(event);
    drag.node.fx = point.x;
    drag.node.fy = point.y;
    drag.moved ||= Math.hypot(
      event.clientX - drag.startX,
      event.clientY - drag.startY,
    ) > 4;
  };

  const onPointerUp = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const drag = pointerRef.current;
    if (!drag) {
      return;
    }
    drag.node.fx = null;
    drag.node.fy = null;
    simulationRef.current?.alphaTarget(0);
    if (!drag.moved) {
      selectRef.current(drag.node.id);
    }
    pointerRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  return (
    <canvas
      ref={canvasRef}
      className="block h-full w-full touch-none"
      style={{ height }}
      role="img"
      aria-label="Interactive word graph. Drag nodes to rearrange; select a word from the accessible board list to inspect it."
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    />
  );
}
