"""Main application"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from par_treasure_hunt import __application_title__, __version__

app = typer.Typer()
console = Console(stderr=True)


def parse_mermaid_graph(file_path: Path) -> dict[str, list[str]]:
    """Parse a Mermaid graph file and return an adjacency list.

    Args:
        file_path: Path to the Mermaid graph file

    Returns:
        Dict mapping node names to lists of adjacent nodes
    """
    # Initialize the adjacency list
    graph: dict[str, list[str]] = defaultdict(list)

    try:
        content = file_path.read_text(encoding="utf-8")

        # Extract the graph content (skip the mermaid wrapper if present)
        if "```mermaid" in content:
            match = re.search(r"```mermaid\s*\n(.*?)\n\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)

        # Find all the edges in the graph
        # Look for various Mermaid edge patterns
        # - Basic connections: "NodeA --- NodeB" or "NodeA --> NodeB"
        # - Labeled nodes: "A[Label A]<-->B[Label B]"
        edge_pattern = re.compile(r"(\w+)(?:\[[^\]]*\])?\s*(?:<?>?-+|<->|<-->)\s*(\w+)(?:\[[^\]]*\])?")

        for line in content.split("\n"):
            # Skip lines that don't define edges
            if "graph" in line.lower() or not line.strip() or line.strip().startswith("%"):
                continue

            match = edge_pattern.search(line)
            if match:
                node1, node2 = match.groups()

                # Add edges in both directions (for undirected graphs)
                graph[node1].append(node2)
                graph[node2].append(node1)

        return dict(graph)
    except Exception as e:
        console.print(f"[red]Error parsing Mermaid graph file: {e}[/red]")
        return {}


def compute_distance_matrix(graph: dict[str, list[str]]) -> dict[str, int]:
    """Compute distances between all pairs of nodes in the graph using BFS.

    Args:
        graph: Adjacency list representation of the graph

    Returns:
        Dictionary mapping "NodeA--NodeB" to their shortest path distance
    """
    distance_matrix: dict[str, int] = {}

    # For each node, run BFS to find distances to all other nodes
    for start_node in graph:
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start_node, 0)])  # (node, distance)

        while queue:
            node, distance = queue.popleft()

            if node in visited:
                continue

            visited.add(node)

            # Add this distance to the matrix
            key = f"{start_node}--{node}"
            distance_matrix[key] = distance

            # Enqueue adjacent nodes
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))

    # Add an entry for each node to itself with distance 0
    for node in graph:
        distance_matrix[f"{node}--{node}"] = 0

    return distance_matrix


def load_distance_matrix(
    locations_file: Path,
    mermaid_graph_file: Path | None = None,
    debug: bool = False,
) -> tuple[dict[str, int], bool]:
    """Compute the distance matrix from a Mermaid graph file.

    Args:
        locations_file: Path to the locations JSON file
        mermaid_graph_file: Path to a Mermaid graph file to compute distances from (optional)
        debug: Whether to print debug information

    Returns:
        Tuple of (distance matrix, whether to use distance optimization)
    """
    # No dist_matrix_file option anymore, we only use Mermaid graph files

    # Try to compute from mermaid_graph_file if provided
    if mermaid_graph_file is not None:
        try:
            console.print(f"Computing distance matrix from Mermaid graph: {mermaid_graph_file}")

            # Parse the graph and compute distances
            graph = parse_mermaid_graph(mermaid_graph_file)
            if not graph:
                console.print("[yellow]Warning: Mermaid graph is empty or could not be parsed[/yellow]")
                return {}, False

            # Compute the distance matrix
            computed_matrix = compute_distance_matrix(graph)

            # Convert format to match room_dist_matrix format
            # Node names in Mermaid use underscores, while room names use spaces
            room_dist_matrix = {}
            for key, distance in computed_matrix.items():
                nodes = key.split("--")
                room1 = nodes[0].replace("_", " ")
                room2 = nodes[1].replace("_", " ")
                room_dist_matrix[f"{room1}--{room2}"] = distance

            console.print(f"Successfully computed distances for {len(room_dist_matrix)} room pairs")
            return room_dist_matrix, True
        except Exception as e:
            console.print(f"[yellow]Warning: Error computing distances from Mermaid graph: {e}[/yellow]")

    # We no longer use JSON distance matrix files

    # Look for a mermaid file with derived name
    try:
        derived_mermaid_path = locations_file.parent / f"{locations_file.stem}_graph.mermaid"
        console.print(f"Looking for Mermaid graph at derived path: {derived_mermaid_path}")

        if derived_mermaid_path.exists():
            # Parse the graph and compute distances
            graph = parse_mermaid_graph(derived_mermaid_path)
            if not graph:
                console.print("[yellow]Warning: Derived Mermaid graph is empty or could not be parsed[/yellow]")
                return {}, False

            # Compute the distance matrix
            computed_matrix = compute_distance_matrix(graph)

            # Convert format to match room_dist_matrix format
            room_dist_matrix = {}
            for key, distance in computed_matrix.items():
                nodes = key.split("--")
                room1 = nodes[0].replace("_", " ")
                room2 = nodes[1].replace("_", " ")
                room_dist_matrix[f"{room1}--{room2}"] = distance

            console.print(
                f"Successfully computed distances for {len(room_dist_matrix)} room pairs from derived Mermaid graph"
            )
            return room_dist_matrix, True
    except Exception as e:
        console.print(f"[yellow]Warning: Error with derived Mermaid graph: {e}[/yellow]")

    # If we got here, we couldn't find/compute a distance matrix
    console.print("[yellow]Falling back to random location selection[/yellow]")
    return {}, False


def generate_treasure_hunt(
    num_places: int,
    first: dict[str, str] | None = None,
    last: dict[str, str] | None = None,
    randomness: float = 0.3,
    debug: bool = False,
    locations_file: Path | str | None = None,
    mermaid_graph_file: Path | str | None = None,
):
    """Generates a treasure hunt with maximized distances between locations.

    Args:
        num_places: Number of places to include in the hunt
        first: Starting location dictionary with room_name and place (optional, random if not provided)
        last: Ending location dictionary with room_name and place (optional, random if not provided)
        randomness: Factor of randomness (0.0-1.0) to introduce in distance calculations
        debug: Enable debug output
        locations_file: Path to JSON file containing locations (optional)
        mermaid_graph_file: Path to Mermaid graph file to compute distances from (optional)
    """
    # Set default file paths if not provided
    if locations_file is None:
        locations_file = Path("locations.json")
    elif isinstance(locations_file, str):
        locations_file = Path(locations_file).resolve()

    # Convert path strings to Path objects if needed
    if isinstance(mermaid_graph_file, str):
        mermaid_graph_file = Path(mermaid_graph_file).resolve()

    # Load locations file
    try:
        console.print(f"Loading locations from: {locations_file}")
        file_locations: list[dict] = json.loads(locations_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        console.print(f"[red]Error: Failed to load locations file: {e}[/red]")
        return

    # Compute the distance matrix from Mermaid graph
    room_dist_matrix, use_distance_optimization = load_distance_matrix(
        locations_file=locations_file,
        mermaid_graph_file=mermaid_graph_file,
        debug=debug,
    )
    locations: list[dict] = []
    for loc in file_locations:
        for place in loc["places"]:
            locations.append({"room_name": loc["room_name"], "place": place})

    console.print(f"{len(locations)} loaded from file")

    # Filter out locations with empty places
    locations = [loc for loc in locations if loc["place"]]

    # If first location not specified, choose a random one
    if first is None:
        if not locations:
            console.print("[red]Error: No locations available[/red]")
            return
        first = random.choice(locations)
        console.print(f"Randomly selected starting location: {first['room_name']} - {first['place']}")

    # Filter out starting point from available locations
    locations = [
        loc for loc in locations if not (loc["room_name"] == first["room_name"] and loc["place"] == first["place"])
    ]
    console.print(f"{len(locations) + 1} after filter")

    # Pre-shuffle to ensure some baseline randomness
    random.shuffle(locations)

    # If last location is specified, remove it from general locations
    final_treasure_location = None
    if last is not None:
        # Remove the specified last location from available locations to avoid duplication
        locations = [
            loc for loc in locations if not (loc["room_name"] == last["room_name"] and loc["place"] == last["place"])
        ]
        final_treasure_location = last
        console.print(f"Using specified final location: {last['room_name']} - {last['place']}")

    # Limit to available places if num_places is too large
    num_places = min(num_places, len(locations))

    # Create optimized path based on distance with randomness
    ordered_locations: list[dict] = []
    current_location = first
    available_locations = locations.copy()

    if use_distance_optimization:
        console.print("Starting pathfinding with distance optimization")
    else:
        console.print("Starting pathfinding with random selection (no distance matrix available)")

    while len(ordered_locations) < num_places and available_locations:
        # If we're using distance optimization
        if use_distance_optimization:
            # Calculate distances from current location to all available locations
            distances: list[tuple[float, dict]] = []

            for loc in available_locations:
                # Get distance from distance matrix
                distance_key = f"{current_location['room_name']}--{loc['room_name']}"
                base_distance = room_dist_matrix.get(distance_key, 1)

                # Add randomness to the distance calculation
                # Higher values get more priority (we're maximizing distance)
                randomized_distance = base_distance * (1 + random.uniform(-randomness, randomness))
                distances.append((randomized_distance, loc))

            # Sort by distance (descending) - use first element (distance) of each tuple
            distances.sort(key=lambda x: x[0], reverse=True)

            # Select the location with maximum distance
            next_location = distances[0][1]
        else:
            # If no distance matrix, just select randomly
            next_location = random.choice(available_locations)
            console.print("[yellow]Using completely random location selection[/yellow]")

        ordered_locations.append(next_location)
        available_locations.remove(next_location)
        current_location = next_location

    # Use the ordered locations
    locations = ordered_locations
    console.print(f"{len(locations) + 1} locations in final path")

    # Calculate max length for padding
    max_room_len = 0
    for location in locations:
        max_room_len = max(max_room_len, len(f"#00 {location['place']}"))
    max_room_len += 2  # add padding

    # Group items by room
    rooms: dict[str, list[str]] = {}
    current_item: dict[str, str] = first
    i: int = 1

    # Get the unique room names (will update if we have a final location)
    room_names: list[str] = list({first["room_name"]} | {room["room_name"] for room in locations})

    # Process each location in the hunt
    for location in locations:
        item: str = (
            f"#{i} {current_item['place']}".ljust(max_room_len) + f" * {location['room_name']} - {location['place']}"
        )
        if current_item["room_name"] not in rooms:
            rooms[current_item["room_name"]] = []
        rooms[current_item["room_name"]].append(item + (" !! X !!" if i == len(locations) else ""))

        current_item = location
        i += 1

    # Use the specified final location for the treasure if provided
    if final_treasure_location is not None:
        treasure_location = final_treasure_location
        console.print(
            f"Placing treasure at specified location: {treasure_location['room_name']} - {treasure_location['place']}"
        )
    else:
        # Otherwise use the last location from our path
        treasure_location = current_item
        console.print(
            f"Placing treasure at path-determined location: {treasure_location['room_name']} - {treasure_location['place']}"
        )

    # Add treasure location to room names if needed
    if treasure_location["room_name"] not in room_names:
        room_names.append(treasure_location["room_name"])

    # Add treasure location
    if treasure_location["room_name"] not in rooms:
        rooms[treasure_location["room_name"]] = []
    rooms[treasure_location["room_name"]].append(f"!!! {treasure_location['place']} TREASURE!!!")

    console.print("[green]Print the following and cut into strips. Keep the strips grouped by room for easier deployment.")
    # Print the treasure hunt
    for room_name in room_names:
        if room_name in rooms:
            print(f"*** {room_name} ***")
            for item in rooms[room_name]:
                print(f"   {item}")


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        print(f"{__application_title__}: {__version__}")
        raise typer.Exit()


# pylint: disable=too-many-arguments,too-many-statements,too-many-branches
@app.command()
def main(
    num_places: Annotated[int, typer.Option("--num-places", "-n", help="Number of places to use")] = 150,
    randomness: Annotated[float, typer.Option("--randomness", "-r", help="Randomness factor (0.0-1.0)")] = 0.3,
    locations_file: Annotated[
        str | None, typer.Option("--locations-file", "-l", help="Path to JSON file with locations")
    ] = None,
    mermaid_graph_file: Annotated[
        str | None, typer.Option("--mermaid-graph", "-m", help="Path to Mermaid graph file to compute distances from")
    ] = None,
    first_room: Annotated[
        str | None, typer.Option("--first-room", "-fr", help="Room name for the starting location")
    ] = None,
    first_place: Annotated[
        str | None, typer.Option("--first-place", "-fp", help="Place name for the starting location")
    ] = None,
    last_room: Annotated[
        str | None, typer.Option("--last-room", "-lr", help="Room name for the final treasure location")
    ] = None,
    last_place: Annotated[
        str | None, typer.Option("--last-place", "-lp", help="Place name for the final treasure location")
    ] = None,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Enable debug mode",
        ),
    ] = False,
    version: Annotated[  # pylint: disable=unused-argument
        bool | None,
        typer.Option("--version", "-v", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    """Generates a treasure hunt with maximized distances between locations.

    By default, the hunt will use a random starting location and a random ending location.
    The path between locations will be chosen to maximize the distance from the previous location,
    with some randomness to make the hunt more interesting.

    Optionally specify:
    - Custom locations file with --locations-file
    - Mermaid graph file with --mermaid-graph to compute distances dynamically
    - Starting location with --first-room and --first-place
    - Final treasure location with --last-room and --last-place

    Distance matrix priority order:
    1. Explicit Mermaid graph file (--mermaid-graph)
    2. Automatically derived Mermaid graph file (locations_graph.mermaid)
    3. Fallback to random location selection if no graph file is available
    """
    # Validate first room and place parameters
    first_location = None
    if first_room is not None and first_place is not None:
        first_location = {"room_name": first_room, "place": first_place}
    elif first_room is not None or first_place is not None:
        console.print("[red]Error: Must provide both --first-room and --first-place or neither[/red]")
        raise typer.Exit(1)

    # Validate last room and place parameters
    last_location = None
    if last_room is not None and last_place is not None:
        last_location = {"room_name": last_room, "place": last_place}
    elif last_room is not None or last_place is not None:
        console.print("[red]Error: Must provide both --last-room and --last-place or neither[/red]")
        raise typer.Exit(1)

    generate_treasure_hunt(
        num_places,
        first=first_location,
        last=last_location,
        randomness=randomness,
        debug=debug,
        locations_file=locations_file,
        mermaid_graph_file=mermaid_graph_file,
    )


if __name__ == "__main__":
    app()
