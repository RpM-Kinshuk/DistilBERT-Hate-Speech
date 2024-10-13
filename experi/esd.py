def net_esd_estimator(
        net=None,
        EVALS_THRESH=0.00001,
        bins=100,
        fix_fingers='xmin_mid',
        xmin_pos=2,
        conv_norm=0.5, 
        filter_zeros=False,
        use_sliding_window=False,
        num_row_samples=100,  # Required for sliding window
        Q_ratio=2.0,  # Required for sliding window
        step_size=10,  # Sliding window step size for variable ops
        sampling_ops_per_dim=None  # For fixed number of operations
    ):
    """Estimator for Empirical Spectral Density (ESD) and Alpha parameter for Conv2D layers."""
    
    results = {
        'alpha': [],
        'spectral_norm': [],
        'D': [],
        'longname': [],
        'eigs': [],
        'norm': [],
        'alphahat': [],
        'eigs_num': []
    }

    print("======================================")
    print(f"fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    print(f"use_sliding_window: {use_sliding_window}, num_row_samples: {num_row_samples}, Q_ratio: {Q_ratio}, step_size: {step_size}, sampling_ops_per_dim: {sampling_ops_per_dim}")
    print("======================================")
    
    device = next(net.parameters()).device  # type: ignore

    for name, m in net.named_modules(): # type: ignore
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            matrix = m.weight.data.clone().to(device)
            
            # Conv2d weight slicing using conv2D_Wmats method from WW
            if isinstance(m, nn.Conv2d):
                # Use the conv2D_Wmats method to slice the Conv2D weight tensor
                Wmats, N, M, rf = conv2D_Wmats(matrix, channels=CHANNELS.UNKNOWN)
            else:  # Linear layers
                Wmats = [matrix.float()]

            all_eigs = []

            for W in Wmats:
                # Apply sliding window sampling or regular ESD computation
                if use_sliding_window:
                    if sampling_ops_per_dim is not None:
                        eigs = fixed_number_of_sampling_ops(
                            W, 
                            num_row_samples=num_row_samples, 
                            Q_ratio=Q_ratio, 
                            num_sampling_ops_per_dimension=sampling_ops_per_dim, 
                        )
                    else:
                        eigs = matrix_size_dependent_number_of_sampling_ops(
                            W, 
                            num_row_samples=num_row_samples, 
                            Q_ratio=Q_ratio, 
                            step_size=step_size,
                        )
                else:
                    # Regular ESD: compute eigenvalues of W
                    eigs = torch.square(torch.linalg.svdvals(W).flatten())
                
                if not isinstance(eigs, torch.Tensor):
                    eigs = torch.tensor(eigs, device=device)

                all_eigs.append(eigs)
            
            # Concatenate and sort all eigenvalues from the slices
            all_eigs = torch.cat(all_eigs)
            all_eigs = torch.sort(all_eigs).values

            spectral_norm = all_eigs[-1].item()
            fnorm = torch.sum(all_eigs).item()

            # Filter based on threshold
            nz_eigs = all_eigs[all_eigs > EVALS_THRESH] if filter_zeros else all_eigs
            if len(nz_eigs) == 0:
                nz_eigs = all_eigs
            N = len(nz_eigs)
            log_nz_eigs = torch.log(nz_eigs)

            # Alpha and D calculations (from before)
            if fix_fingers == 'xmin_mid':
                i = N // xmin_pos
                xmin = nz_eigs[i]
                n = float(N - i)
                seq = torch.arange(n, device=device)
                final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                final_D = torch.max(torch.abs(1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n))
            else:
                alphas = torch.zeros(N-1, device=device)
                Ds = torch.ones(N-1, device=device)
                if fix_fingers == 'xmin_peak':
                    hist_nz_eigs = torch.log10(nz_eigs)
                    min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                    counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e) # type: ignore
                    boundaries = torch.linspace(min_e, max_e, bins + 1) # type: ignore
                    ih = torch.argmax(counts)
                    xmin2 = 10 ** boundaries[ih]
                    xmin_min = torch.log10(0.95 * xmin2)
                    xmin_max = 1.5 * xmin2
                
                for i, xmin in enumerate(nz_eigs[:-1]):
                    if fix_fingers == 'xmin_peak':
                        if xmin < xmin_min:
                            continue
                        if xmin > xmin_max:
                            break
                    n = float(N - i)
                    seq = torch.arange(n, device=device)
                    alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                    alphas[i] = alpha
                    if alpha > 1:
                        Ds[i] = torch.max(torch.abs(1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n))
                
                min_D_index = torch.argmin(Ds)
                final_alpha = alphas[min_D_index]
                final_D = Ds[min_D_index]

            # Store results
            final_alpha = final_alpha.item()  # type: ignore
            final_D = final_D.item()  # type: ignore
            final_alphahat = final_alpha * math.log10(spectral_norm)

            results['spectral_norm'].append(spectral_norm)
            results['alphahat'].append(final_alphahat)
            results['norm'].append(fnorm)
            results['alpha'].append(final_alpha)
            results['D'].append(final_D)
            results['longname'].append(name)
            results['eigs'].append(nz_eigs.cpu().numpy())
            results['eigs_num'].append(len(nz_eigs))

    return results