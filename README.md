# HSR Restaurant Service Robot (Lucy)

---


A ROS Noetic implementation of an autonomous restaurant service robot built on the **Toyota Human Support Robot (HSR)**. This repository contains a ROS Noetic implementation of a restaurant service robot built on the Toyota Human Support Robot (HSR). The system integrates perception, navigation, and human-robot interaction to detect customers, navigate to them, take orders, and deliver items.

---

## Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the System](#running-the-system)
- [ROS Topics & Services](#ros-topics--services)
- [Module Reference](#module-reference)
- [Configuration Parameters](#configuration-parameters)
- [Testing](#testing)
- [Development Team](#development-team)

---

## System Overview

The system follows a service pipeline:

1. Detect: Camera frames are processed using YOLO pose estimation to identify hand-raise gestures.
2. Queue: Detected customers are stored in a FIFO queue with basic prioritisation. FIFO queue with spatial deduplication
3. Navigate: A* is used for global planning over the occupancy grid. A* was chosen over move_base in Solution 2 to allow tighter control over clearance and path smoothing.
4. Order: Speech is transcribed using Whisper; a local LLM extracts structured items.
5. Confirm - extracted items are read back; customer confirms or corrects
6. Collect - Robot returns to the bar, announces the order, and waits for staff to load the tray
7. Deliver - Robot navigates back to the customer and hands over the items
8. Return - Robot returns home and becomes ready for the next customer


---
## Two Navigation Solutions
 
This package has **two complete, independently runnable solutions**. Both share the same perception stack. They differ in how navigation is handled and whether HRI is included.
 
---
 
### Solution 1 - Custom Perception + move_base (Person Following)
 
Detects customers via hand-raise gestures and navigates to them using the ROS move_base stack. This solution does not include order-taking functionality.
This solution focuses on person-following using the standard ROS navigation stack.

**Stack:**
 
```
perception_node  ->  /queue_person_pose
        |
        V
person_queue_manager  ->  /person_pose
        |
        V
behavior_loop  ->  /goal_pose  ->  move_base (global + local planners)
```
 
**Key nodes:**
 
| Node | Role |
|---|---|
| `perception_node.py` | YOLO gesture detection, publishes customer map pose |
| `person_queue_manager.py` | Queues multiple detected customers |
| `behavior_loop.py` | State machine: samples goal positions around the customer, validates them via `move_base/make_plan`, sends the best to `move_base`, handles stuck detection and iterative convergence |
 
**How `behavior_loop` uses move_base:**
- Calls `/move_base/make_plan` to validate candidate goals around the customer before committing
- Publishes the chosen goal to `/goal_pose` which is forwarded to `move_base`
- Calls `/move_base/clear_costmaps` when stuck
- Monitors `/navigation/status` for `goal_reached` / `no_path` events
- States: `IDLE -> MOVING_TO_PERSON -> WAITING -> MOVING_HOME`
 
**Launch:**
 
```bash
# Terminal 1 - SLAM + Navigation
roslaunch hsr_perception lucy_demo.launch

# Terminal 2 - Perception 
roslaunch hsr_perception perception.launch
```
 
---

### Solution 2 - Custom Perception + Custom Navigation + HRI (Full Restaurant Service)
 
Full restaurant service loop - detects customers, navigates using a custom A\* planner and potential-field local planner, takes orders via speech, and delivers them. No dependency on `move_base`. This is a 2nd package we delivered which tags along with our perception stack and own navigation stack built from scratch.
 
**Stack:**
 
```
perception_node  ->  /queue_person_pose
        |
        V
person_queue_manager  ->  /person_pose
        |
        V
navigation_manager  (A* global planner + WaypointExtractor)
        |
        V  /waypoint_goal
field_planner  (potential field local planner)
        |
        V  /hsrb/command_velocity
        |
        V
restaurant_hri_node  ->  whisper_service_node  ->  restaurant_service_node (Ollama LLM)
```
 
**Key nodes:**
 
| Node | Role |
|---|---|
| `perception_node.py` | YOLO gesture detection, publishes customer map pose |
| `person_queue_manager.py` | Queues multiple detected customers |
| `navigation_manager.py` | A\* path planning, waypoint extraction, full service state machine |
| `field_planner.py` | Potential-field local planner, publishes velocity commands directly |
| `restaurant_hri_node.py` | Order-taking dialogue (STT -> LLM -> confirm -> save) |
| `whisper_service_node.py` | Local Whisper STT service |
| `restaurant_service_node.py` | Ollama LLM order extraction service |
 
**Launch:**
 
```bash
# Terminal 1 - Full navigation + HRI stack
roslaunch hsr_navigation navigation.launch
 
# Terminal 2 - Perception
roslaunch hsr_perception perception.launch
```
 
---
 
### Solution Comparison
 
| | Solution 1 | Solution 2 |
|---|---|---|
| **Global planner** | move_base (NavFn / DWA) | Custom A\* with clearance & smoothing |
| **Local planner** | move_base DWA | Custom potential-field (`field_planner.py`) |
| **Goal validation** | `move_base/make_plan` service | Costmap clearance check + nearest free cell |
| **Order taking** | Whisper STT + Ollama LLM |
| **Delivery workflow** | Full bar -> customer loop |
| **move_base dependency** |  Required | Not required |
| **Velocity output** | move_base -> cmd_vel | `field_planner` -> `/hsrb/command_velocity` |
 
---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PERCEPTION LAYER                         │
│  Camera -> perception_node -> gesture_detector -> /queue_person_pose │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────V─────────────────────────────────┐
│                       QUEUE MANAGEMENT                          │
│           person_queue_manager -> /person_pose                  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────V─────────────────────────────────┐
│                     NAVIGATION LAYER                            │
│  navigation_manager (A* + waypoints) -> field_planner (PF)      │
│                    /waypoint_goal                               │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────V─────────────────────────────────┐
│                         HRI LAYER                               │
│  restaurant_hri_node <-> whisper_service <-> restaurant_service │
│               (STT)          (Whisper)    (Ollama LLM)          │
└─────────────────────────────────────────────────────────────────┘
```

**Flag bus** (`/flag_in` / `/flag_out`) synchronises state transitions across all layers.

---

## Repository Structure

```
src/lucy/hsr_ws/src/
├── hsr_navigation/
│   ├── config/
│   │   └── nav_stack_config.rviz
│   ├── launch/
│   │   ├── navigation.launch
│   │   └── person_queue_manager.launch
│   └── scripts/
│       ├── a_star_planner.py          # Grid-based A* with clearance & smoothing
│       ├── field_planner.py           # Potential-field local planner (ROS node)
│       ├── navigation_manager.py      # High-level service workflow state machine
│       ├── person_queue_manager.py    # Multi-customer FIFO queue
│       └── waypoints_extractor.py     # Dense-path -> sparse waypoint reduction
│
├── hsr_perception/
│   ├── launch/
│   │   ├── perception.launch
│   │   └── lucy_demo.launch
│   └── scripts/
│       ├── behavior_loop.py           # Person-following state machine
│       ├── gesture_detector.py        # YOLO pose -> hand-raise detection
│       └── perception_node.py         # ROS wrapper, camera -> gesture events
│
└── llm_server/
    ├── scripts/
    │   ├── restaurant_hri_node.py     # Order-taking dialogue manager
    │   ├── restaurant_service_node.py # Ollama LLM order extraction service
    │   └── whisper_service_node.py    # Whisper STT ROS service
    └── srv/
        └── ExtractOrder.srv
```

---

## Prerequisites

| Requirement | Version | Notes |
| --- | --- | --- | 
| Ubuntu | 20.04 | Required by ROS Noetic |
| ROS | Noetic | Full desktop install recommended |
| Python | 3.8+ | with Ubuntu 20.04 |
| Ollama | latest | [ollama.com](https://ollama.com) |
| CUDA (optional) | 11.x+ | Speeds up YOLO inference |

### ROS apt packages

```bash
sudo apt update
sudo apt install -y \
  ros-noetic-cv-bridge \
  ros-noetic-tf2-ros \
  ros-noetic-tf2-geometry-msgs \
  ros-noetic-slam-toolbox \
  ros-noetic-move-base \
  ros-noetic-rviz \
  ros-noetic-message-filters \
  python3-rosdep \
  python3-catkin-tools
```

---

## Installation

###  Install Python dependencies

```bash
pip install -r requirements.txt
```

### Populate the catkin workspace

```bash
git clone <https://git.inf.h-brs.de/sdp-ws2025/robocup-at-home/person-following-assistive-robot.git> 
cd ~/person-following-assistive-robot/src/lucy/hsr_ws/
catkin_make
source devel/setup.bash
```

> **Note on PyAudio:** if the install fails, first run:
> ```bash
> sudo apt install -y portaudio19-dev
> pip install pyaudio
> ```

### Install Ollama and pull the LLM model

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model used by restaurant_service_node.py
ollama pull granite3.1-moe:1b
```

### Download the YOLO weights

The perception node expects `yolo11n-pose.pt`. It downloads automatically on first run:

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
```

Or place it manually in `src/hsr_perception/scripts/`.

### 6. Build the workspace

```bash
cd ~/src/lucy/hsr_ws/
catkin_make
source devel/setup.bash
```

Add the source line to your shell profile so you don't need to repeat it:

```bash
echo "source ~/hsr_ws/devel/setup.bash" >> ~/.bashrc
```

---

## Running the System


> **Every terminal** below assumes the workspace is sourced:
> ```bash
> source ~/catkin_ws/devel/setup.bash
> ```

### Quick start - full demo

Launches SLAM, navigation, perception, RViz, and all HRI services in one command this for the `move_base` demo:

```bash
roslaunch hsr_perception lucy_demo.launch
```

---

### Step-by-step launch

Open a separate terminal for each step.

**Terminal 1 - ROS core**

```bash
cd ~/hsr_ws/
source ../connect_to_lucy.sh  # this ensures the roscore 
```

**Terminal 2 - SLAM ,mapping, navigation stack**

```bash
rosnode kill /map_server    # Kill the map server so that we ensure it starts fresh with mapping.
roslaunch hsr_navigation navigation.launch
```

This starts: `navigation_manager`, `person_queue_manager`, `field_planner`, RViz, `whisper_service_node`, `restaurant_service_node`, and `restaurant_hri_node`.

**Terminal on Lucy (or on local machine) - Perception**

```bash
roslaunch hsr_perception perception.launch
```

This starts: `perception_node` (camera + YOLO gesture detection) and `speech_relay`.

---

### Verify the system is running

Check all expected nodes are alive:

```bash
rosnode list
```

Expected nodes include:

```
/navigation_manager
/person_queue_manager
/field_based_planner
/person_perception_node
/whisper_service_node
/order_extraction_service
/restaurant_hri_node
/rviz
```

Monitor the flag bus:

```bash
rostopic echo /flag_out
```

Manually inject a customer to test the full pipeline without the camera:

```bash
rostopic pub /queue_person_pose geometry_msgs/PoseStamped \
  "header: {frame_id: 'map'} pose: {position: {x: 1.0, y: 1.0, z: 0.0} orientation: {w: 1.0}}" \
  --once
```

---

### Microphone setup

The Whisper service defaults to device index `8`. Find your device index:

```bash
python3 -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"
```

Update `device_index` in `whisper_service_node.py` to match your hardware.

---

## ROS Topics & Services
 
### Topics
 
| Topic | Type | Solution | Description |
|---|---|---|---|
| `/queue_person_pose` | `PoseStamped` | Both | Raw detected customer position from perception |
| `/person_pose` | `PoseStamped` | Both | Next customer dispatched by queue manager |
| `/goal_pose` | `PoseStamped` | Sol.1 | Goal sent by behavior_loop to move_base |
| `/waypoint_goal` | `PoseStamped` | Sol.2 | Waypoint sent by navigation_manager to field_planner |
| `/flag` | `String` | Sol.1 | `home_reached`, `customer_reached` from behavior_loop |
| `/flag_in` | `String` | Sol.2 | HRI → navigation (`order_taken`, `items_ready`) |
| `/flag_out` | `String` | Sol.2 | Navigation → HRI (`customer_reached`, `bar_reached`, `delivery_complete`, `home_reached`) |
| `/say` | `String` | Sol.2 | Text-to-speech output |
| `/condition_record` | `Bool` | Sol.2 | Microphone on/off control |
| `/navigation/status` | `String` | Both | `goal_reached`, `no_path`, `planning_failed` |
| `/scan` | `LaserScan` | Sol.2 | LiDAR for obstacle avoidance |
| `/depth_scan` | `LaserScan` | Sol.2 | Depth camera virtual scan |
| `/map` | `OccupancyGrid` | Both | Global occupancy map from SLAM |
| `/move_base/global_costmap/costmap` | `OccupancyGrid` | Sol.1 | move_base costmap |
| `/detection/debug_image` | `Image` | Both | Annotated YOLO frame for RViz |
 
### Services
 
| Service | Type | Solution | Provider |
|---|---|---|---|
| `/move_base/make_plan` | `GetPlan` | Sol.1 | move_base — goal validation |
| `/move_base/clear_costmaps` | `Empty` | Sol.1 | move_base — costmap reset |
| `speech_recognize` | `Trigger` | Sol.2 | `whisper_service_node` |
| `extractOrder` | `ExtractOrder` | Sol.2 | `restaurant_service_node` |
 
---

## Module Reference

### `gesture_detector.py`

YOLO11n-pose based hand-raise detector. Iterates all detected persons per frame (up to 5), checks wrist/elbow positions relative to shoulders, and applies temporal hysteresis (2 consecutive frames) to confirm a gesture.

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `TRIGGER_THRESHOLD` | `2` | Consecutive frames required to confirm gesture |
| `KEYPOINT_CONF_THRESH` | `0.5` | Minimum YOLO keypoint confidence |
| `model_path` | `yolo11n-pose.pt` | YOLO weights file |

**Returns per frame:** `gesture_detected`, `nose_coords`, `person_orientation`, `annotated_frame`

---
### `behavior_loop.py`

This is a 2nd package we delivered which tags along with our pereption stack and uses `move_base` package to navigate as well as planning. 

It takes the online async as the mapping tool and uses the move_base nav stack to plan from the current position to the goal which triggered byt the perception node.

---

### `navigation_manager.py`

Service workflow state machine. States: `IDLE -> TAKING_ORDER -> RETURNING_FROM_ORDER -> WAITING_FOR_ITEMS -> DELIVERING -> RETURNING_HOME`.

Uses `AStarPlanner` for global path planning and `WaypointExtractor` to reduce the dense path to sparse waypoints. Publishes each waypoint sequentially to `/waypoint_goal`. Saves the outbound path and replays it reversed on the return trip.

**Key ROS params:**

| Param | Default | Description |
|---|---|---|
| `~robot_radius` | `0.45` | Inflation radius (m) for obstacle map |
| `~waypoint_distance` | `1.0` | Min distance (m) between extracted waypoints |
| `~goal_tolerance` | `0.5` | Distance (m) to declare goal reached |
| `~replan_distance` | `2.0` | Trigger replan when this close to current sub-goal |

---

### `a_star_planner.py`

Standard A\* on a 2D occupancy grid with:
- **Clearance map** (distance transform) - penalises paths close to walls
- **Direction penalty** - favours straight trajectories
- **Gradient-descent smoothing** - 5-iteration post-processing pass

**Key methods:** `plan(start_world, goal_world)` -> list of `(x, y)`, `smooth_path()`, `get_nearest_free_cell()`

---

### `field_planner.py`

Potential-field local planner implemented as a ROS node. Combines attractive force toward goal with repulsive forces from LiDAR (`/scan`) and depth scan (`/depth_scan`). Includes stuck detection and a recovery manoeuvre (reverse + rotate).

**Key ROS params:**

| Param | Default | Description |
|---|---|---|
| `~ka` | `0.75` | Attractive force gain |
| `~kr` | `4.0` | Repulsive force gain |
| `~p_0` | `0.4` | Obstacle influence radius (m) |
| `~stop_threshold` | `0.35` | Distance (m) to declare arrival |
| `~stuck_time_threshold` | `2.0` | Seconds without movement to trigger recovery |

---

### `person_queue_manager.py`

FIFO deque for detected customers. Deduplicates by spatial proximity (`duplication_distance_threshold = 1.5 m`). Waits 6 seconds after first detection before dispatching. On `delivery_complete`, skips returning home if the queue is non-empty.

---

### `waypoints_extractor.py`

Framework-agnostic utility. Three extraction methods: `distance`, `angle`, `combined`.

---

### `restaurant_hri_node.py`

Dialogue manager. On `customer_reached`: listen -> LLM extract -> confirm -> save JSON -> publish `order_taken`. On `bar_reached`: announce order, wait for `"items placed"` -> publish `items_ready`.

---

### `restaurant_service_node.py`

ROS service wrapper around Ollama. Uses `granite3.1-moe:1b` with temperature `0.1`. Returns dish names as a comma-separated string.

---

### `whisper_service_node.py`

`Trigger` service. Opens microphone (device index 8), records up to 15 s, transcribes with local `base.en` Whisper model, returns transcript string.

---

## Configuration Parameters
 
| Node | Solution | Param | Default |
|---|---|---|---|
| `behavior_loop` | Sol.1 | `ACCEPTABLE_DIST` | `1.4 m` |
| `behavior_loop` | Sol.1 | `GLOBAL_TIMEOUT` | `120 s` |
| `behavior_loop` | Sol.1 | `STUCK_TIMEOUT` | `25 s` |
| `navigation_manager` | Sol.2 | `~robot_radius` | `0.45` |
| `navigation_manager` | Sol.2 | `~waypoint_distance` | `1.0` |
| `navigation_manager` | Sol.2 | `~goal_tolerance` | `0.5` |
| `navigation_manager` | Sol.2 | `~replan_distance` | `2.0` |
| `field_planner` | Sol.2 | `~ka` | `0.75` |
| `field_planner` | Sol.2 | `~kr` | `4.0` |
| `field_planner` | Sol.2 | `~p_0` | `0.4` |
| `field_planner` | Sol.2 | `~stop_threshold` | `0.35` |
| `field_planner` | Sol.2 | `~stuck_time_threshold` | `2.0` |
| `field_planner` | Sol.2 | `~recovery_duration` | `2.5` |
| `person_queue_manager` | Both | `~max_queue_size` | `10` |
| `person_queue_manager` | Both | `~duplication_distance` | `1.5` |
| `perception_node` | Both | `~process_every_n` | `3` |
| `perception_node` | Both | `~depth_min_m` | `0.3` |
| `perception_node` | Both | `~depth_max_m` | `10.0` |
| `restaurant_service_node` | Sol.2 | `~ollama_model` | `granite3.1-moe:1b` |
| `restaurant_hri_node` | Sol.2 | `~json_dir` | `/tmp/hri_orders` |

---


## Testing

# Individual modules
pytest tests/test_astar_planner.py
pytest tests/test_gesture_detector.py
pytest tests/test_behavior_loop.py
pytest tests/test_field_planner.py
pytest tests/test_navigation_manager.py
pytest tests/test_perception_node.py
```

CI is configured in `.gitlab-ci.yml`.

##  Development Team

  Name                 Responsibility
  -------------------- ------------------------------
  **Jannen Thyriar**   Perception & Vision Pipeline, NavigationFSM & Behavior Architecture 
  **Nikhil Ravi**      Behavior trees 
  **Al Shafi**         Navigation & SLAM Systems  

------------------------------------------------------------------------

## 📘 Documentation Status

This README serves as a high-level overview.\
A full technical report, API documentation, and deployment guide will be created upon deployment.

