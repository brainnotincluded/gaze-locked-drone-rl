#!/usr/bin/env python3
"""
Webots Supervisor controller that broadcasts pedestrian position via TCP.
Place in: Webots controller directory or run as external controller.

The drone inference script connects to this server on localhost:9999
and receives [x, y, z] pedestrian position in world coordinates.
"""

import socket
import json
import threading
import time


def run_server(pedestrian_node, port=9999):
    """Broadcast pedestrian position over TCP."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(5)
    server.settimeout(1.0)
    print(f"[Supervisor] Position server listening on port {port}")

    while True:
        try:
            conn, addr = server.accept()
            threading.Thread(
                target=handle_client,
                args=(conn, pedestrian_node),
                daemon=True,
            ).start()
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[Supervisor] Server error: {e}")
            break

    server.close()


def handle_client(conn, pedestrian_node):
    """Stream position to connected client."""
    try:
        while True:
            pos = pedestrian_node.getPosition()
            data = json.dumps({"x": pos[0], "y": pos[1], "z": pos[2]}) + "\n"
            conn.sendall(data.encode())
            time.sleep(0.02)  # 50Hz
    except Exception:
        pass
    finally:
        conn.close()


# ── Webots entry point ──────────────────────────────────────────────────────
try:
    from controller import Supervisor

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    # Get pedestrian node by DEF name
    pedestrian = robot.getFromDef("PEDESTRIAN")
    if pedestrian is None:
        print("[Supervisor] ERROR: PEDESTRIAN DEF not found in world!")
        print(
            "             Make sure to add 'DEF PEDESTRIAN' before Pedestrian { in .wbt"
        )
    else:
        print(f"[Supervisor] Found PEDESTRIAN node")
        pos = pedestrian.getPosition()
        print(f"[Supervisor] Initial position: {pos}")

        # Start TCP server in background thread
        threading.Thread(target=run_server, args=(pedestrian,), daemon=True).start()

        # Run simulation
        while robot.step(timestep) != -1:
            pass  # Webots drives the loop

except ImportError:
    print("[Supervisor] Not running inside Webots - standalone test mode")
    import sys

    sys.exit(0)
