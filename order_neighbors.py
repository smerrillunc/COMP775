def order_neighbors(input, queries, queries_norm, idx, proj, outi, angles):
    batch_size, m_q, n = queries.size()
    k = idx.size(2)

    for batch_index in range(batch_size):
        queries_batch = queries[batch_index]
        queries_norm_batch = queries_norm[batch_index]
        idx_batch = idx[batch_index]
        angles_batch = angles[batch_index]
        outi_batch = outi[batch_index]
        input_batch = input[batch_index]
        proj_batch = proj[batch_index]

        # 1. Project points on a plane
        x_p = input_batch[idx_batch, : , 0]
        y_p = input_batch[idx_batch, :, 1]
        z_p = input_batch[idx_batch, :, 2]
        n_x = queries_norm_batch[:, 0]
        n_y = queries_norm_batch[:, 1]
        n_z = queries_norm_batch[:, 2]
        x_q = queries_batch[:, 0]
        y_q = queries_batch[:, 1]
        z_q = queries_batch[:, 2]

        d = ((x_p - x_q.unsqueeze(1)) * n_x.unsqueeze(0) +
             (y_p - y_q.unsqueeze(1)) * n_y.unsqueeze(0) +
             (z_p - z_q.unsqueeze(1)) * n_z.unsqueeze(0))

        proj_batch[:, :, :, 0] = x_p.unsqueeze(1) - d.unsqueeze(3) * n_x.unsqueeze(0)
        proj_batch[:, :, :, 1] = y_p.unsqueeze(1) - d.unsqueeze(3) * n_y.unsqueeze(0)
        proj_batch[:, :, :, 2] = z_p.unsqueeze(1) - d.unsqueeze(3) * n_z.unsqueeze(0)

        # 2. Calculate angles
        curvature_x = proj_batch[:, 0, :, 0]
        curvature_y = proj_batch[:, 0, :, 1]
        curvature_z = proj_batch[:, 0, :, 2]
        BCx = curvature_x - x_q
        BCy = curvature_y - y_q
        BCz = curvature_z - z_q

        for j in range(1, k):
            x_p = proj_batch[:, j, :, 0]
            y_p = proj_batch[:, j, :, 1]
            z_p = proj_batch[:, j, :, 2]

            ACx = x_p - x_q
            ACy = y_p - y_q
            ACz = z_p - z_q

            cross_pr = torch.cross(torch.stack([ACx, ACy, ACz], dim=-1),
                                   torch.stack([BCx, BCy, BCz], dim=-1))

            det = (cross_pr[:, 0] * n_x + cross_pr[:, 1] * n_y + cross_pr[:, 2] * n_z).unsqueeze(1)
            cos_theta = ((ACx * BCx + ACy * BCy + ACz * BCz) /
                         (torch.norm(torch.stack([ACx, ACy, ACz], dim=-1), dim=-1) *
                          torch.norm(torch.stack([BCx, BCy, BCz], dim=-1), dim=-1))).unsqueeze(1)

            mask = det < 0
            angles_batch[:, j - 1] = torch.where(mask, -cos_theta - 2, cos_theta).squeeze()

        # 3. Sort neighbors
        _, indices = torch.sort(angles_batch[:, :k - 1], dim=1, descending=True)

        for i in range(m_q):
            outi_batch[i, :k - 1] = idx_batch[i, indices[i]]
            proj_batch[i, :k - 1] = proj_batch[i, indices[i]]

