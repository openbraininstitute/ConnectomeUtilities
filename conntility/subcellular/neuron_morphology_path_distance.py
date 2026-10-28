# SPDX-License-Identifier: Apache-2.0
import morphio
import numpy


class MorphologyPathDistanceCalculator(object):
    """
    A helper to quickly calculate path distances along the morphology of a neuron.

    Relies on pre-calculating section-tree metadata (parent links, root path lengths,
    and within-section segment offsets) for a single morphology. This is only really
    worth the cost when you want to look up distances between hundreds of points on a
    single morphology. For smaller use cases (or cases where the distances are to be
    calculated on many different morphologies) other code may be faster.

    NOTE: This class makes the strong assumption that the first section in the morphology
    is a direct descendant of the soma.
    """

    def __init__(self, morphology: morphio.Morphology):
        """
        Args:
          morphology (morphio.Morphology): The morphology you want to calculate path distances for.
        """
        # MorphIO returns a fresh dict/lists on each connectivity access — snapshot once.
        self._connectivity = morphology.connectivity
        assert 0 in self._connectivity[-1], "This class requires the first section to be connected to the soma!"
        self.m = morphology
        self._n_sections = len(morphology.section_types)
        self.offset = self.segment_offset_table(morphology)
        (
            self._parent,
            self._root_dist,
            self._sec_len,
            self._tin,
            self._tout,
            self._up,
        ) = self.__initialize_tree__()

    @staticmethod
    def segment_offset_table(m):
        """
        Calculate for each segment in each section the distance from the root of the section to the
        root of the segment. Since different sections have different numbers of segments, all lists
        of offsets are padded with NaN and stacked to a 2d array.

        Built from MorphIO's flat ``points`` / ``section_offsets`` arrays (no Section wrappers).

        Args:
          m (morphio.Morphology): The morphogy in question

        Returns:
          offset_table (numpy.array): Table of segment offsets. To be indexed as
          offset_table[sec_id][seg_id].
          Note: sec_id is base-0, i.e. index 0 is NOT the soma. The soma is not listed
          at all (assumed to be just a point).
        """
        points = numpy.asarray(m.points, dtype=float)
        section_offsets = numpy.asarray(m.section_offsets, dtype=numpy.int64)
        n_sec = len(section_offsets) - 1
        n_pts_per_sec = numpy.diff(section_offsets)
        max_pts = int(n_pts_per_sec.max()) if n_sec else 0
        table = numpy.full((n_sec, max_pts), numpy.nan, dtype=float)
        for i in range(n_sec):
            start = int(section_offsets[i])
            stop = int(section_offsets[i + 1])
            table[i, 0] = 0.0
            n_pts = stop - start
            if n_pts > 1:
                seg_lengths = numpy.sqrt(
                    numpy.sum(numpy.diff(points[start:stop], axis=0) ** 2, axis=1)
                )
                table[i, 1:n_pts] = numpy.cumsum(seg_lengths)
        return table

    def __initialize_tree__(self):
        """
        Build O(n) section-tree tables used for on-demand path distances.

        Returns:
            parent (numpy.array): Parent section id for each section, -1 for roots.
            root_dist (numpy.array): Path length from the soma to the first point of each section.
            sec_len (numpy.array): Euclidean length of each section.
            tin, tout (numpy.array): DFS discovery/finish times for O(1) ancestor tests.
            up (numpy.array): Binary-lifting parents, shape (log n, n).
        """
        n = self._n_sections
        connectivity = self._connectivity

        parent = numpy.full(n, -1, dtype=numpy.int32)
        root_dist = numpy.zeros(n, dtype=float)
        sec_len = numpy.nanmax(numpy.asarray(self.offset, dtype=float), axis=1)
        tin = numpy.full(n, -1, dtype=numpy.int32)
        tout = numpy.full(n, -1, dtype=numpy.int32)

        # connectivity is parent_id -> child section ids; -1 is the soma.
        # Prefer this over m.sections / root_sections (those allocate Section wrappers).
        for p, children in connectivity.items():
            if p < 0:
                continue
            for child in children:
                parent[child] = p

        # Roots stay at root_dist 0 (soma treated as a point). Children attach at the
        # end of their parent, so root_dist[child] = root_dist[parent] + sec_len[parent].
        # Also assign DFS in/out times for ancestor queries.
        timer = 0
        # iterative DFS with explicit enter/exit using a stack of (node, child_index)
        stack = [(root, 0) for root in reversed(connectivity.get(-1, ()))]
        while stack:
            node, idx = stack.pop()
            children = connectivity.get(node, ())
            if idx == 0:
                tin[node] = timer
                timer += 1
                if parent[node] >= 0:
                    root_dist[node] = root_dist[parent[node]] + sec_len[parent[node]]
            if idx < len(children):
                stack.append((node, idx + 1))
                stack.append((children[idx], 0))
            else:
                tout[node] = timer
                timer += 1

        log_n = max(1, n.bit_length())
        up = numpy.full((log_n, n), -1, dtype=numpy.int32)
        up[0] = parent
        for k in range(1, log_n):
            prev = up[k - 1]
            mask = prev >= 0
            up[k, mask] = prev[prev[mask]]

        return parent, root_dist, sec_len, tin, tout, up

    def within_section_offsets(self, locs,
                               str_section_id="afferent_section_id",
                               str_segment_id="afferent_segment_id",
                               str_offset="afferent_segment_offset"):
        """
        Calculate the offset of points within a section.

        Args:
          locs (pandas.DataFrame): dataframe where columns specify the section id, segment id and within-segment offset
          of points on the neuron morphology.
          str_section_id (default="afferent_section_id"): The name of the column holding the section id.
          str_segment_id (default="afferent_segment_id"): The name of the column holding the segment id.
          str_offset (defaults="afferent_segment_offset"): The name of the column holding the within-segment offset.

        Note: The default column names are set to what bluepysnap returns when asked for afferent synapses.
        Also, the section id is expected to be base-1 indexed. That is, index 0 is the soma,
        index 1 is the first "proper" section.
        As path distance along the soma is a tricky concept, the function will return 0.0
        for all entries on the soma!

        Returns:
          offsets (numpy.array): The within-section offsets in the same order as the input.
        """
        offsets = self.offset[locs[str_section_id].values - 1, locs[str_segment_id]]
        assert not numpy.any(numpy.isnan(offsets))
        offsets = offsets + locs[str_offset].values
        offsets[locs[str_section_id].values == 0] = 0.0
        return offsets

    def path_distances(self, locs_from, locs_to=None,
                       str_section_id="afferent_section_id",
                       str_segment_id="afferent_segment_id",
                       str_offset="afferent_segment_offset",
                       same_section_only=False):
        """
        Calculate path distances. Either pairwise, or between two sets of points.
        Args:
          locs_from (pandas.DataFrame): Locations of points on the morphology represented
          by this object to calculate the distances from. A dataframe where columns specify
          the section id, segment id and within-segment offset of points on the neuron morphology.
          locs_to (pandas.DataFrame, default=None): Locations to calculate the distances to. If none, locs_from is used.
          str_section_id (default="afferent_section_id"): The name of the column holding the section id.
          str_segment_id (default="afferent_segment_id"): The name of the column holding the segment id.
          str_offset (defaults="afferent_segment_offset"): The name of the column holding the within-segment offset.
          same_section_only (bool, default=False): If True, set values for all pairs NOT on the same section to NaN.

          Note: The default column names are set to what bluepysnap returns when asked for afferent synapses.
          Also, the section id is expected to be base-1 indexed. That is, index 0 is the soma,
          index 1 is the first "proper" section.
          This is also compatible with how bluepysnap returns the information.
          Since the concept of path distance along the soma is tricky, this method assumes
          that the soma is a point. Consequently, for all locations on the soma, the values
          will be as if the location was at the root of a root section.

        Returns:
          dist (numpy.array): Array of pairwise path distances.
        """
        if locs_to is None:
            locs_to = locs_from

        o_from = self.within_section_offsets(locs_from, str_section_id=str_section_id,
                                             str_segment_id=str_segment_id, str_offset=str_offset)
        o_to = self.within_section_offsets(locs_to, str_section_id=str_section_id,
                                           str_segment_id=str_segment_id, str_offset=str_offset)
        if same_section_only:
            # +1 for soma id 0; use cached section count (not len(m.sections)).
            same_section = numpy.eye(self._n_sections + 1, dtype=bool)
            same_section = same_section[numpy.ix_(locs_from[str_section_id], locs_to[str_section_id])]
            dist = numpy.nan * numpy.ones(same_section.shape, dtype=float)
            i, j = numpy.nonzero(same_section)
            dist[i, j] = numpy.abs(o_from[i] - o_to[j])
            return dist

        sec_idx_fr = locs_from[str_section_id].to_numpy(dtype=numpy.int32, copy=True) - 1
        sec_idx_to = locs_to[str_section_id].to_numpy(dtype=numpy.int32, copy=True) - 1
        # Since we test above that the first section is rooted at the soma, we can use that
        # section in place of the soma. This makes sense, because we assume the soma to be a
        # point (and consequently, within_section_offset) returns 0.0 for all locations on the soma.
        sec_idx_fr[sec_idx_fr < 0] = 0
        sec_idx_to[sec_idx_to < 0] = 0

        return self._path_distances_vectorized(sec_idx_fr, sec_idx_to, o_from, o_to)

    def _path_distances_vectorized(self, sec_idx_fr, sec_idx_to, o_from, o_to):
        """Vectorized pairwise path distances using O(1) ancestor tests and binary lifting."""
        tin = self._tin
        tout = self._tout
        rd = self._root_dist
        sec_len = self._sec_len
        parent = self._parent
        up = self._up

        n_from = len(sec_idx_fr)
        n_to = len(sec_idx_to)
        sf = numpy.broadcast_to(sec_idx_fr[:, None], (n_from, n_to)).copy()
        st = numpy.broadcast_to(sec_idx_to[None, :], (n_from, n_to)).copy()

        same = sf == st
        # rel 1: to is ancestor of from; rel 2: from is ancestor of to
        anc_to = (tin[st] <= tin[sf]) & (tout[sf] <= tout[st]) & ~same
        anc_fr = (tin[sf] <= tin[st]) & (tout[st] <= tout[sf]) & ~same
        sib = ~(same | anc_to | anc_fr)

        rd_f = rd[sf]
        rd_t = rd[st]
        root_D = numpy.zeros((n_from, n_to), dtype=float)
        root_D[anc_to] = (rd_f - rd_t)[anc_to]
        root_D[anc_fr] = (rd_t - rd_f)[anc_fr]

        # Sibling / cousin / other-root pairs: LCA via binary lifting from `from`, then
        # branch point at the end of the LCA section (or soma depth 0).
        a = sf.copy()
        for k in range(up.shape[0] - 1, -1, -1):
            ak = numpy.where(a >= 0, up[k, numpy.maximum(a, 0)], -1)
            is_anc = (
                (ak >= 0)
                & (tin[numpy.maximum(ak, 0)] <= tin[st])
                & (tout[st] <= tout[numpy.maximum(ak, 0)])
            )
            need = sib & (a >= 0) & (ak >= 0) & ~is_anc
            a[need] = ak[need]
        lca = numpy.where(a >= 0, parent[a], -1)
        branch = numpy.zeros((n_from, n_to), dtype=float)
        valid = sib & (lca >= 0)
        branch[valid] = (rd[lca] + sec_len[lca])[valid]
        root_D[sib] = (rd_f + rd_t - 2.0 * branch)[sib]

        of = o_from[:, None]
        ot = o_to[None, :]
        dist = numpy.empty((n_from, n_to), dtype=float)
        dist[same] = numpy.abs(of - ot)[same]
        dist[anc_to] = (root_D + of - ot)[anc_to]
        dist[anc_fr] = (root_D - of + ot)[anc_fr]
        dist[sib] = (root_D + of + ot)[sib]
        return dist

    @staticmethod
    def __nnd__(D, _idxx):
        if len(_idxx) <= 1:
            return numpy.nan
        return numpy.nanmin(D[numpy.ix_(_idxx, _idxx)], axis=0)

    @staticmethod
    def __rnd_sampler__(locs, prop_col, n_rnd):
        import pandas
        if prop_col is None:
            tbl = locs["index"].reset_index(drop=True)
        else:
            tbl = locs.set_index(prop_col)["index"]
        tbl_idx = pandas.Series(numpy.unique(tbl.index))

        def sampler(N):
            for _ in range(n_rnd):
                yield tbl[tbl_idx.sample(N)].values
        return sampler

    def nearest_neighbor_distances(self, locs, normalize=False,
                                   normalize_preserve=None,
                                   normalize_n=10,
                                   str_section_id="afferent_section_id",
                                   str_segment_id="afferent_segment_id",
                                   str_offset="afferent_segment_offset",
                                   same_section_only=False):
        """
        Analyzes the nearest neighbor distances between sets of dendritic locations and optionally
        normalizes them against random controls.

        Args:
          locs (pandas.DataFrame): Locations of points on the morphology represented by this object.
          For details, see MorphologyPathDistanceCalculator.path_distances. Additionally, must have
          an index that defines a grouping (see below).
          str_section_id (str, optional): See MorphologyPathDistanceCalculator.path_distances.
          str_segment_id (str, optional): See MorphologyPathDistanceCalculator.path_distances.
          str_offset (str, optional): See MorphologyPathDistanceCalculator.path_distances.
          same_section_only (bool, default: False): See MorphologyPathDistanceCalculator.path_distances.
          normalize (bool, default: False): Whether to normalize the results by converting to a
          z-score with respect to a random control
          normalize_preserve (optional): Name of a column in locs that defines a grouping.
          When normalizing and this is specified, then locations belonging to the same group will be kept
          together in the random controls. Ignored if normalize=False
          normalize_n (int, default: 10): How many random controls to generate for normalization.
          Ignored if normalize=False.

        Explaining this analysis in the context of locations of synapses on dendritic trees and their
        (potential) clustering.
        The DataFrame locs defines the locations of all afferent synapses onto the morphology in terms of
        section id, segment id and offset. It has an index (can be MultiIndex) that groups them into any
        number of groups by their unique values. This can be the name of the source region, identifier of
        a structural or functional cluster, or simply a bool specifying whether the synapse is of interest
        or not. The analysis then investigates whether the synapses belonging to the same group are
        clustered, i.e. closer together than expected, or not, and to what degree.

        res = M.nearest_neighbor_distances(locs, normalize=False) simply returns a DataFrame with the
        nearest neighbor distances of synapses in each group. It has one row per unique value of the input
        index, and values are lists with the nearest neighbor distances of all synapses associated with
        that index.

        res = M.nearest_neighbor_distances(locs, normalize=True) converts the distances into z-scores
        by comparing to a random control. The control is generated by randomly sampling the same number of
        locations from locs. That is, it compares the distances in each group to the distances in random
        groups of the same sizes.

        res = M.nearest_neighor_distances(locs, normalize=True, normalize_preserve="some_property")
        slightly adjusts how the random control is generates. The value of normalize_preserve must be the
        name of a column in locs. Then, in the generation of the controls, locations that have the same
        value in that column will always be sampled together. For example, the column could hold an
        identifier of the presynaptic neuron. Then you can use tis feature to test the clustering of
        inputs from a neuronal assembly while taking into account that inputs from the same presynaptic
        neuron are already likely to be slightly clustered.

        res = M.nearest_neighbor_distance(..., normalize_n=25) controls how many random controls are
        generated for normalization.

        Examples
        """
        D = self.path_distances(locs, str_section_id=str_section_id,
                                str_segment_id=str_segment_id, str_offset=str_offset,
                                same_section_only=same_section_only)
        numpy.fill_diagonal(D, numpy.nan)
        index_names = list(locs.index.names)
        assert "index" not in index_names
        locs = locs.reset_index().reset_index()
        data = locs.groupby(index_names)["index"].apply(lambda _s: self.__nnd__(D, _s))
        if not normalize:
            return data

        sampler = self.__rnd_sampler__(locs, normalize_preserve, normalize_n)
        if normalize_preserve is None:
            normalize_preserve = "index"
        counts = locs.groupby(index_names)[normalize_preserve].apply(lambda x: len(x.drop_duplicates()))
        ctrl = counts.apply(lambda c: numpy.hstack([self.__nnd__(D, _s) for _s in sampler(c)]))
        return data.combine(ctrl, lambda _d, _c: (numpy.nanmean(_d) - numpy.nanmean(_c)) / numpy.nanstd(_c))
