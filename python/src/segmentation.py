import cv2
import numpy as np

from split_merged_tips import split_merged_tips

_NEIGHBORS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def load_image(path):
    """Charge en niveaux de gris."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def to_binary(gray):
    """Binarise via Otsu : trait=255 (blanc), fond=0 (noir)."""
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def segment_states(binary_image, relative_min_area=0.005, min_area_cap=500):
    """
    Detecte les etats : composantes du fond circulaires (l'interieur d'un
    cercle est un trou du fond).

    Le seuil d'aire est relatif a l'image mais plafonne : les cercles
    d'etats ont une taille a peu pres fixe (rendu Graphviz), alors que
    l'image grandit avec l'automate. Sans plafond, les grandes images
    filtrent tous les etats.
    """
    image_height, image_width = binary_image.shape
    min_area = min(image_width * image_height * relative_min_area,
                   min_area_cap)

    background_image = cv2.bitwise_not(binary_image)

    n_components, labels_image, component_stats, component_centroids = (
        cv2.connectedComponentsWithStats(background_image, connectivity=8)
    )
    state_candidates = []
    for component_index in range(1, n_components):
        bounding_x = component_stats[component_index, cv2.CC_STAT_LEFT]
        bounding_y = component_stats[component_index, cv2.CC_STAT_TOP]
        bounding_width = component_stats[component_index, cv2.CC_STAT_WIDTH]
        bounding_height = component_stats[component_index, cv2.CC_STAT_HEIGHT]
        component_area = component_stats[component_index, cv2.CC_STAT_AREA]

        touches_left = bounding_x == 0
        touches_top = bounding_y == 0
        touches_right = bounding_x + bounding_width == image_width
        touches_bottom = bounding_y + bounding_height == image_height
        if touches_left or touches_top or touches_right or touches_bottom:
            continue

        if component_area < min_area:
            continue

        component_mask = (labels_image == component_index).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            continue

        contour = contours[0]
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * component_area / (perimeter ** 2)
        if circularity < 0.85:
            continue

        approximate_radius = (bounding_width + bounding_height) / 4
        centroid_x = component_centroids[component_index, 0]
        centroid_y = component_centroids[component_index, 1]

        bx, by = int(bounding_x), int(bounding_y)
        bw, bh = int(bounding_width), int(bounding_height)

        name_crop = binary_image[by:by+bh, bx:bx+bw].copy()
        local_center = (int(centroid_x) - bx, int(centroid_y) - by)
        disk = np.zeros_like(name_crop)
        cv2.circle(disk, local_center, int(approximate_radius) - 2, 255, -1)
        name_crop = cv2.bitwise_and(name_crop, disk)
        state_candidates.append({
            "center_x": float(centroid_x),
            "center_y": float(centroid_y),
            "radius": float(approximate_radius),
            "outer_radius": float(approximate_radius),
            "accepting": False,
            "area": int(component_area),
            "name_crop": name_crop,
        })

    return state_candidates


def remove_states_from_img(binary_image, states, padding=3):
    """
    Efface les etats. Utilise outer_radius si dispo (acceptants),
    sinon radius. Permet d'effacer proprement les double-cercles.
    """
    cleaned_image = binary_image.copy()
    for state in states:
        center = (int(state["center_x"]), int(state["center_y"]))
        base_radius = state.get("outer_radius", state["radius"])
        erase_radius = int(base_radius) + padding
        cv2.circle(cleaned_image, center, erase_radius, color=0, thickness=-1)
    return cleaned_image


def merge_states_and_acceptants(states_first_pass, residual_circles, center_tolerance=10):
    for state in states_first_pass:
        state["accepting"] = False
        for residual in residual_circles:
            dx = state["center_x"] - residual["center_x"]
            dy = state["center_y"] - residual["center_y"]
            distance = (dx * dx + dy * dy) ** 0.5
            if distance <= center_tolerance:
                state["accepting"] = True
                state["outer_radius"] = residual["radius"]
                break
    return states_first_pass


def find_triangles(img_cleaned):
    image_height, image_width = img_cleaned.shape
    k = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(img_cleaned, cv2.MORPH_OPEN, k)
    n_components, labels_image, component_stats, component_centroids = (
        cv2.connectedComponentsWithStats(opened, connectivity=8)
    )
    tips = []
    for component_index in range(1, n_components):
        bounding_x = component_stats[component_index, cv2.CC_STAT_LEFT]
        bounding_y = component_stats[component_index, cv2.CC_STAT_TOP]
        bounding_width = component_stats[component_index, cv2.CC_STAT_WIDTH]
        bounding_height = component_stats[component_index, cv2.CC_STAT_HEIGHT]
        component_area = component_stats[component_index, cv2.CC_STAT_AREA]

        touches_left = bounding_x == 0
        touches_top = bounding_y == 0
        touches_right = bounding_x + bounding_width == image_width
        touches_bottom = bounding_y + bounding_height == image_height
        if touches_left or touches_top or touches_right or touches_bottom:
            continue

        if component_area < 6:
            continue

        component_mask = (labels_image == component_index).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            continue
        perimeter = cv2.arcLength(contours[0], closed=True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * component_area / (perimeter ** 2)
        if circularity > 0.97:
            continue

        tips.append({
            "centroid": (float(component_centroids[component_index, 0]),
                         float(component_centroids[component_index, 1])),
            "area": int(component_area),
            "bbox": (
                int(bounding_x),
                int(bounding_y),
                int(bounding_width),
                int(bounding_height),
            ),
            "pixels": np.where(labels_image == component_index),
        })

    return split_merged_tips(tips)


def analyze_tip(tip_pixels, centroid, states):
    """
    Determine l'apex (point du blob le plus proche d'un etat), la direction
    (centroide -> centre de l'etat destination) et l'index de destination.

    tip_pixels : (ys, xs) coordonnees GLOBALES des pixels du blob.
    """
    cx, cy = centroid
    ys_global, xs_global = tip_pixels

    best_apex = None
    best_distance = float("inf")
    best_dest = None
    best_state_center = None

    for state_index, state in enumerate(states):
        scx, scy = state["center_x"], state["center_y"]
        distances = np.sqrt((xs_global - scx) ** 2 + (ys_global - scy) ** 2)
        min_idx = np.argmin(distances)
        if distances[min_idx] < best_distance:
            best_distance = distances[min_idx]
            best_apex = (float(xs_global[min_idx]), float(ys_global[min_idx]))
            best_dest = state_index
            best_state_center = (scx, scy)

    if best_apex is None:
        # aucun etat detecte dans l'image : pas de destination possible
        return None, (0.0, 0.0), None

    if best_state_center is not None:
        direction_x = best_state_center[0] - cx
        direction_y = best_state_center[1] - cy
    else:
        direction_x = best_apex[0] - cx
        direction_y = best_apex[1] - cy

    norm = (direction_x ** 2 + direction_y ** 2) ** 0.5
    direction = (direction_x / norm, direction_y / norm) if norm > 0 else (0.0, 0.0)

    return best_apex, direction, best_dest




def reached_state(x, y, states, step, origin_dest_index, min_steps_before_return):
    """Retourne l'index de l'etat atteint par (x, y), ou None."""
    for idx, s in enumerate(states):
        if idx == origin_dest_index and step < min_steps_before_return:
            continue
        r = s.get("outer_radius", s["radius"]) + 8
        if (x - s["center_x"])**2 + (y - s["center_y"])**2 <= r*r:
            return idx
    return None




def find_state_in_direction(origin, direction, states, exclude=None,
                            cos_min=0.5, max_distance=None):
    """Etat le plus proche dans le cone defini par direction (fallback)."""
    candidates = []
    for index, state in enumerate(states):
        if index == exclude:
            continue
        delta_x = state["center_x"] - origin[0]
        delta_y = state["center_y"] - origin[1]
        norm_delta = (delta_x ** 2 + delta_y ** 2) ** 0.5
        if norm_delta == 0:
            continue
        cos_theta = (delta_x * direction[0] + delta_y * direction[1]) / norm_delta
        if cos_theta < cos_min:
            continue
        distance_squared = delta_x ** 2 + delta_y ** 2
        if max_distance and distance_squared > max_distance ** 2:
            continue
        candidates.append((index, distance_squared))
    if not candidates:
        return None
    return min(candidates, key=lambda c: c[1])[0]



def isolate_labels(img_clean_final, tips, arrows):

    n_components, labels_image, stats, centroids = (
        cv2.connectedComponentsWithStats(img_clean_final, connectivity=8)
    )

    arrow_component_ids = set()

    for arrow in arrows:
        for (px, py) in arrow["chemin"]:
            if 0 <= py < labels_image.shape[0] and 0 <= px < labels_image.shape[1]:
                label_id = labels_image[py, px]
                if label_id != 0:
                    arrow_component_ids.add(label_id)

    for tip in tips:
        ys, xs = tip["pixels"]
        if len(ys) > 0:
            label_id = labels_image[ys[0], xs[0]]
            if label_id != 0:
                arrow_component_ids.add(label_id)

    img_labels = img_clean_final.copy()
    for component_id in arrow_component_ids:
        img_labels[labels_image == component_id] = 0

    return img_labels

def extract_label_crop(img_labels, bbox, padding=2, target_size=32):
    """Decoupe la lettre dans sa bbox, centre et normalise en target_size."""
    bx, by, bw, bh = bbox
    h, w = img_labels.shape
    y0 = max(0, by - padding)
    y1 = min(h, by + bh + padding)
    x0 = max(0, bx - padding)
    x1 = min(w, bx + bw + padding)
    raw = img_labels[y0:y1, x0:x1]

    ch, cw = raw.shape
    if ch == 0 or cw == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)
    side = max(ch, cw)
    square = np.zeros((side, side), dtype=np.uint8)
    yo = (side - ch) // 2
    xo = (side - cw) // 2
    square[yo:yo + ch, xo:xo + cw] = raw
    return cv2.resize(square, (target_size, target_size),
                      interpolation=cv2.INTER_NEAREST)

def segment_name_characters(name_crop, min_area=5, target_size=32):
    """
    Decoupe le nom d'un etat (name_crop) en caracteres : composantes
    connexes triees de gauche a droite, chacune extraite via son propre
    masque (le padding de extract_label_crop n'attrape ainsi pas les
    pixels du caractere voisin). Retourne une liste de crops normalises.
    """
    n_components, labels_image, stats, _ = cv2.connectedComponentsWithStats(
        name_crop, connectivity=8
    )

    kept = [i for i in range(1, n_components)
            if stats[i, cv2.CC_STAT_AREA] >= min_area]
    kept.sort(key=lambda i: (stats[i, cv2.CC_STAT_LEFT],
                             stats[i, cv2.CC_STAT_TOP]))

    crops = []
    for component_index in kept:
        bbox = (
            int(stats[component_index, cv2.CC_STAT_LEFT]),
            int(stats[component_index, cv2.CC_STAT_TOP]),
            int(stats[component_index, cv2.CC_STAT_WIDTH]),
            int(stats[component_index, cv2.CC_STAT_HEIGHT]),
        )
        mask = (labels_image == component_index).astype(np.uint8) * 255
        crops.append(extract_label_crop(mask, bbox, padding=2,
                                        target_size=target_size))
    return crops


def assign_labels_to_arrows(img_labels, arrows, min_area=10):
    for arrow in arrows:
        arrow["labels"] = []

    n_components, labels_image, stats, centroids = (
        cv2.connectedComponentsWithStats(img_labels, connectivity=8)
    )

    for component_index in range(1, n_components):
        area = stats[component_index, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cx = centroids[component_index, 0]
        cy = centroids[component_index, 1]

        best_arrow = None
        best_distance = float("inf")
        for arrow in arrows:
            for (px, py) in arrow["chemin"]:
                d = (px - cx) ** 2 + (py - cy) ** 2
                if d < best_distance:
                    best_distance = d
                    best_arrow = arrow


        if best_arrow is not None:
            bbox = (
                int(stats[component_index, cv2.CC_STAT_LEFT]),
                int(stats[component_index, cv2.CC_STAT_TOP]),
                int(stats[component_index, cv2.CC_STAT_WIDTH]),
                int(stats[component_index, cv2.CC_STAT_HEIGHT]),
            )

            best_arrow["labels"].append({
                "centroid": (float(cx), float(cy)),
                "bbox": bbox,
                "crop": extract_label_crop(img_labels, bbox, padding=2),
                "pixels": np.where(labels_image == component_index),
                "symbol": None,
            })

    return arrows

def _branch_candidates(cx, cy, img, visited, direction, min_align, gap_deg):
    """
    Regroupe les voisins blancs non visites, alignes avec direction, en amas
    angulaires distincts. Retourne un representant (le mieux aligne) par amas.
    1 representant -> suivi simple ; 2+ -> bifurcation.
    Un trace perpendiculaire (croisement) a un alignement faible et est ecarte.
    """
    height, width = img.shape
    cands = []  # (angle, (nx, ny), align)
    for dx, dy in _NEIGHBORS:
        nx, ny = cx + dx, cy + dy
        if not (0 <= ny < height and 0 <= nx < width):
            continue
        if img[ny, nx] == 0 or (nx, ny) in visited:
            continue
        n = (dx * dx + dy * dy) ** 0.5
        if n == 0:
            continue
        align = (dx * direction[0] + dy * direction[1]) / n
        if align < min_align:          # ecarte perpendiculaire / arriere
            continue
        cands.append((np.arctan2(dy, dx), (nx, ny), align))

    if not cands:
        return []

    cands.sort(key=lambda c: c[0])
    gap = gap_deg * np.pi / 180.0

    reps = []  # (alignement, representant) par amas
    cluster_best_align = -2.0
    cluster_best_pt = None
    prev_angle = cands[0][0]
    open_cluster = False
    for angle, pt, align in cands:
        if open_cluster and angle - prev_angle > gap:   # nouvel amas
            reps.append((cluster_best_align, cluster_best_pt))
            cluster_best_align = -2.0
        if align > cluster_best_align:
            cluster_best_align = align
            cluster_best_pt = pt
        prev_angle = angle
        open_cluster = True
    if open_cluster:
        reps.append((cluster_best_align, cluster_best_pt))
    # reps[0] sert de chemin unique hors bifurcation : le mieux aligne d'abord
    reps.sort(key=lambda r: -r[0])
    return [pt for _align, pt in reps]


def _lookahead_score(img, visited, start, direction, n_steps):
    """
    Mini-marche gloutonne de n_steps depuis start (direction figee) ;
    retourne l'alignement du deplacement net avec direction. Sert a
    departager les sorties d'un croisement : le trait qui continue tout
    droit garde un score proche de 1, le trait croise devie et chute.
    """
    height, width = img.shape
    cx, cy = start
    seen = {start}
    for _ in range(n_steps):
        best, best_align = None, None
        for dx, dy in _NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if img[ny, nx] == 0 or (nx, ny) in visited or (nx, ny) in seen:
                continue
            n = (dx * dx + dy * dy) ** 0.5
            align = (dx * direction[0] + dy * direction[1]) / n
            if best_align is None or align > best_align:
                best, best_align = (nx, ny), align
        if best is None:
            break
        seen.add(best)
        cx, cy = best
    vx, vy = cx - start[0], cy - start[1]
    n = (vx * vx + vy * vy) ** 0.5
    if n == 0:
        return -1.0
    return (vx * direction[0] + vy * direction[1]) / n


def follow_line_branches(img, start_pt, initial_dir, states,
                         origin_dest_index=None, max_steps=2000,
                         smoothing_window=5, min_steps_before_return=8,
                         fork_min_align=0.15, fork_gap_deg=20.0, max_branches=4,
                         lookahead_steps=6, fork_lookahead_min=0.6):
    """
    Suit le trace depuis start_pt en detectant les bifurcations (pile explicite,
    visited partage). Retourne une liste de (chemin, source) : une entree par
    branche atteignant un etat, sources dedupliquees.
    """
    height, width = img.shape

    sx, sy = int(start_pt[0]), int(start_pt[1])
    if not (0 <= sy < height and 0 <= sx < width) or img[sy, sx] == 0:
        ys, xs = np.where(img > 0)
        if len(xs) == 0:
            return []
        i = int(np.argmin((xs - start_pt[0]) ** 2 + (ys - start_pt[1]) ** 2))
        sx, sy = int(xs[i]), int(ys[i])

    # pile de frames : (pos, direction, chemin)
    stack = [((sx, sy), initial_dir, [(sx, sy)])]
    visited = {(sx, sy)}

    results = []
    seen_sources = set()
    branches_used = 0

    while stack:
        _pos, direction, chemin = stack.pop()

        for _step in range(max_steps):
            cx, cy = chemin[-1]

            hit = None
            for idx, s in enumerate(states):
                if idx == origin_dest_index and len(chemin) < min_steps_before_return:
                    continue
                r = s.get("outer_radius", s["radius"]) + 8
                if (cx - s["center_x"]) ** 2 + (cy - s["center_y"]) ** 2 <= r * r:
                    hit = idx
                    break
            if hit is not None:
                if hit not in seen_sources:
                    seen_sources.add(hit)
                    results.append((chemin, hit))
                break

            reps = _branch_candidates(cx, cy, img, visited, direction,
                                      fork_min_align, fork_gap_deg)
            if not reps:
                break  # cul-de-sac

            if len(reps) > 1:
                # croisement possible : le look-ahead departage les sorties.
                # On garde la meilleure et celles qui restent bien alignees
                # (vraies bifurcations), on ecarte les traits croises.
                scored = sorted(
                    ((_lookahead_score(img, visited, pt, direction,
                                       lookahead_steps), pt)
                     for pt in reps),
                    key=lambda sp: -sp[0])
                reps = [pt for la, pt in scored
                        if la >= fork_lookahead_min or pt == scored[0][1]]

            if len(reps) == 1 or branches_used >= max_branches:
                nx, ny = reps[0]
                visited.add((nx, ny))
                chemin.append((nx, ny))
                rx, ry = chemin[max(0, len(chemin) - smoothing_window)]
                dx, dy = nx - rx, ny - ry
                n = (dx * dx + dy * dy) ** 0.5
                if n:
                    direction = (dx / n, dy / n)
            else:
                for nx, ny in reps:
                    if (nx, ny) in visited:
                        continue
                    visited.add((nx, ny))
                    dx, dy = nx - cx, ny - cy
                    n = (dx * dx + dy * dy) ** 0.5
                    d = (dx / n, dy / n) if n else direction
                    stack.append(((nx, ny), d, chemin + [(nx, ny)]))
                    branches_used += 1
                break  # la frame courante est remplacee par ses branches

    return results

def build_adjacency_matrix(matrix, states, triangles, img_clean_final,
                           initial_state=None):
    """
    Pour chaque triangle :
      1. analyze_tip -> destination + direction (centroide -> etat dest)
      2. on efface la tete sur une copie, on part de la base du triangle
      3. follow_line suit le trace jusqu'a un etat -> source
         (gere les auto-boucles : source peut == destination)
      4. fallback geometrique (demi-plan) si le suivi echoue
      5. si vraiment aucun etat trouve -> fleche initiale

    La matrice contient, pour chaque transition trouvee, une reference vers
    l'objet arete (le meme que dans `arrows`), ce qui donne acces aux labels
    et crops via matrix[src][dst].

    Returns: (matrix, initial_state_index, arrows)
    """
    initial_state_index = initial_state
    arrows = []
    for triangle in triangles:
        pixels = triangle.get("pixels")
        if pixels is None or len(pixels[0]) == 0:
            continue
        cx, cy = triangle["centroid"]
        apex, direction, dest_index = analyze_tip(pixels, (cx, cy), states)
        if dest_index is None:
            continue
        base_x = 2 * cx - apex[0]
        base_y = 2 * cy - apex[1]
        init_dx = base_x - apex[0]
        init_dy = base_y - apex[1]
        init_norm = (init_dx ** 2 + init_dy ** 2) ** 0.5
        if init_norm > 0:
            init_dir = (init_dx / init_norm, init_dy / init_norm)
        else:
            init_dir = (-direction[0], -direction[1])
        img_suivi = img_clean_final.copy()
        ys, xs = pixels
        img_suivi[ys, xs] = 0

        branches = follow_line_branches(
            img_suivi, (base_x, base_y), init_dir, states,
            origin_dest_index=dest_index
        )

        if not branches:
            opposite = (-direction[0], -direction[1])
            src = find_state_in_direction((cx, cy), opposite, states,
                                          exclude=dest_index)
            if src is None:
                if initial_state_index is None:
                    initial_state_index = dest_index  # fleche initiale
                continue
            branches = [([], src)]

        for chemin, source_index in branches:
            edge = {
                "source": source_index,
                "dest": dest_index,
                "chemin": chemin,
                "tip": triangle,
                "labels": [],
            }
            arrows.append(edge)
            matrix[source_index][dest_index] = edge
        matrix[source_index][dest_index] = edge

    return matrix, initial_state_index, arrows
