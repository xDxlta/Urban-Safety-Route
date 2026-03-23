import osmnx as ox
import networkx as nx

G = ox.load_graphml("zurich_walk.graphml")

# Calculate risk for each edge based on tunnel and highway attributes
for u, v, k, data in G.edges(keys=True, data=True):
    tunnel_value = data.get("tunnel")
    highway = data.get("highway")

    risk = 0

    if tunnel_value in ["yes", "building_passage", "covered"]:
        risk += 1.0

    if highway in ["primary", "secondary", "tertiary", "service"]:
        risk += 0.5

    data["risk"] = risk

# Define a custom weight function that combines length and risk
def safe_weight(u, v, data):
    edge_data = min(data.values(), key=lambda x: x.get("length", float("inf")))
    return edge_data.get("length", 0) + 10000 * edge_data.get("risk", 0)


def get_routes(start, end):
    orig = ox.distance.nearest_nodes(G, start[1], start[0])
    dest = ox.distance.nearest_nodes(G, end[1], end[0])

    route_normal = nx.shortest_path(G, orig, dest, weight="length")
    route_safe = nx.shortest_path(G, orig, dest, weight=safe_weight)

    route_normal_gdf = ox.routing.route_to_gdf(G, route_normal)
    route_safe_gdf = ox.routing.route_to_gdf(G, route_safe)

    return route_normal_gdf, route_safe_gdf


start = (47.3780, 8.5400)
end = (47.3730, 8.5485)

# Get the normal and safe routes as GeoDataFrames
route_normal_gdf, route_safe_gdf = get_routes(start, end)

print(type(route_normal_gdf))
print(type(route_safe_gdf))

print(route_normal_gdf.to_json()[:500])