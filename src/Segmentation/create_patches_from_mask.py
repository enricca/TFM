def create_patches_from_mask(image, mask, patchSize=32, pad=32, depth=1, searchSlices=None):
    """
    Creates all patches.
    Returns list of x, y, z; (center slice, left lower border)
            image data
            label
            nPixels inside
    """
    rois = []
    images = []
    labels = []
    searchSlices = range(len(mask)) if searchSlices is None else searchSlices
    for i in searchSlices:
        # For each voxel, generate a ROI centered there
        if not np.any(mask[i]):
            continue
        xS, yS = np.nonzero(mask[i, :, :])
        xS -= xS % patchSize
        yS -= yS % patchSize
        allPatches = set(zip(xS, yS))
        for x, y in allPatches:
            patch = np.copy(
                # agafem el patch que ens interessa i agafem un contorn per si de cas (padding)
                # potser seria interessant reduir el padding (la quantitat de marge que deixem)
                # ara mateix tenim patches de 96, quan ens interessa el centre de 32 d'aquests
                image[i - depth: i + 1 + depth, x - pad:x + patchSize + pad, y - pad:y + patchSize + pad]
            )
            label = np.copy(
                # quan fem rotacio al fer data augmentation, ens volem assegurar d'estar treballant amb
                # el mateix
                mask[i: i + 1, x - pad: x + patchSize + pad, y - pad:y + patchSize + pad]
            )

            rois.append(np.array([x, y, i]))
            images.append(patch)
            labels.append(label)
    return rois, images, labels