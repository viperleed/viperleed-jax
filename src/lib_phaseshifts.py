import numpy as np
import os
import sys
import scipy
import fortranformat as ff

def readPHASESHIFTS(sl, rp, readfile='PHASESHIFTS', check=True,
                    ignoreEnRange=False):
    """Reads from a PHASESHIFTS file, returns the data as a list of tuples
    (E, enps), where enps is a list of lists, containing one list of values
    (for different L) for each element. Therefore, len(phaseshifts) is the
    number of energies found, len(phaseshifts[0][1]) should match the number
    of elements, and len(phaseshifts[0][1][0]) is the number of different
    values of L in the phaseshift file. The "check" parameters controls
    whether the phaseshifts that were found should be checked against the
    parameters / slab. If it is set to False, then passing "None" as sl and rp
    will work."""
    rf74x10 = ff.FortranRecordReader('10F7.4')
    ri3 = ff.FortranRecordReader('I3')

    # legacy - allow "_PHASESHIFTS"
    if (readfile == 'PHASESHIFTS' and not os.path.isfile('PHASESHIFTS')
            and os.path.isfile('_PHASESHIFTS')):
        logger.info("Found no PHASESHIFTS file, but found legacy file named "
                    "_PHASESHIFTS. Renaming _PHASESHIFTS to PHASESHIFTS.")
        os.rename('_PHASESHIFTS', 'PHASESHIFTS')

    try:
        rf = open(readfile, 'r')
    except FileNotFoundError:
        logger.error("PHASESHIFTS file not found.")
        raise

    filelines = []
    for line in rf:
        filelines.append(line[:-1])
    rf.close()

    try:
        nel = ri3.read(filelines[0])[0]
    except Exception:
        logger.error("Exception while trying to read PHASESHIFTS: could not "
                     "find number of blocks in first line.")
        raise
    phaseshifts = []

    firstline = filelines[0]
    readline = 1
    linesperblock = 0
    while readline < len(filelines):
        if linesperblock:
            en = rf74x10.read(filelines[readline])[0]
            enps = []
            for i in range(0, nel):
                elps = []
                for j in range(0, linesperblock):
                    llist = rf74x10.read(filelines[readline
                                                   + (i*linesperblock)+j+1])
                    llist = [f for f in llist if f is not None]
                    elps.extend(llist)
                enps.append(elps)
            phaseshifts.append((en, enps))
            readline += linesperblock*nel+1
        else:
            # first check how many lines until the next energy:
            lineit = 1
            llist = rf74x10.read(filelines[readline+lineit])
            llist = [f for f in llist if f is not None]
            longestline = len(llist)
            shortestline = longestline
            lastlen = longestline
            cont = True
            while cont:
                lineit += 1
                llist = rf74x10.read(filelines[readline+lineit])
                llist = [f for f in llist if f is not None]
                if len(llist) == 1:
                    if lastlen == 1 or (shortestline > 1
                                        and shortestline < longestline):
                        cont = False  # found next energy
                    else:
                        shortestline = 1
                elif len(llist) != longestline:
                    shortestline = len(llist)
                lastlen = len(llist)
            linesperblock = int((lineit-1)/nel)
            if not linesperblock or (((lineit-1)/nel) - linesperblock != 0.0):
                logger.warning(
                    "Error while trying to read PHASESHIFTS: "
                    "Could not parse file: The number of blocks may not match "
                    "the number given in the first line. A new PHASESHIFTS "
                    "file will be generated.")
                rp.setHaltingLevel(1)
                return ("", [], True, True)
            # don't increase readline -> read the same block again afterwards

    if not check:
        newpsGen, newpsWrite = False, False
    else:
        # check whether the phaseshifts that were found fit the data:
        newpsGen, newpsWrite = True, True
        # recommend that new values should be generated / written
        psblocks = 0
        for el in sl.elements:
            if el in rp.ELEMENT_MIX:
                n = len(rp.ELEMENT_MIX[el])
            else:
                n = 1
            psblocks += n*len([s for s in sl.sitelist if s.el == el])
        # check for MUFTIN parameters:
        muftin = True
        llist = firstline.split()
        if len(llist) >= 6:
            for i in range(1, 5):
                try:
                    float(llist[i])
                except ValueError:
                    muftin = False
        else:
            muftin = False
        if rp.V0_REAL == "default" and not muftin:
            logger.warning(
                "Could not convert first line of PHASESHIFTS file to MUFTIN "
                "parameters. A new PHASESHIFTS file will be generated.")
            rp.setHaltingLevel(1)
        elif len(phaseshifts[0][1]) == psblocks:
            logger.debug("Found "+str(psblocks)+" blocks in PHASESHIFTS "
                         "file, which is consistent with PARAMETERS.")
            newpsGen, newpsWrite = False, False
        elif len(phaseshifts[0][1]) == len(sl.chemelem):
            logger.warning(
                "Found fewer blocks than expected in the "
                "PHASESHIFTS file. However, the number of blocks matches "
                "the number of chemical elements. A new PHASESHIFTS file "
                "will be generated, assuming that each block in the old "
                "file should be used for all atoms of one element.")
            rp.setHaltingLevel(1)
            oldps = phaseshifts[:]
            phaseshifts = []
            for (en, oldenps) in oldps:
                enps = []
                j = 0   # block index in old enps
                for el in sl.elements:
                    if el in rp.ELEMENT_MIX:
                        m = len(rp.ELEMENT_MIX[el])
                    else:
                        m = 1
                    n = len([s for s in sl.sitelist if s.el == el])
                    for i in range(0, m):    # repeat for chemical elements
                        for k in range(0, n):   # repeat for sites
                            enps.append(oldenps[j])
                        j += 1  # count up the block in old enps
                phaseshifts.append((en, enps))
            newpsGen = False
            firstline = str(len(phaseshifts[0][1])).rjust(3) + firstline[3:]
        else:
            logger.warning(
                "PHASESHIFTS file was read but is inconsistent with "
                "PARAMETERS. A new PHASESHIFTS file will be generated.")
            rp.setHaltingLevel(1)

    if check and not ignoreEnRange:
        # check whether energy range is large enough:
        checkfail = False
        er = np.arange(rp.THEO_ENERGIES[0], rp.THEO_ENERGIES[1]+1e-4,
                       rp.THEO_ENERGIES[2])
        psmin = round(phaseshifts[0][0]*27.211396, 2)
        psmax = round(phaseshifts[-1][0]*27.211396, 2)
        if rp.V0_REAL == "default" or type(rp.V0_REAL) == list:
            if type(rp.V0_REAL) == list:
                c = rp.V0_REAL
            else:
                llist = firstline.split()
                c = []
                try:
                    for i in range(0, 4):
                        c.append(float(llist[i+1]))
                except (ValueError, IndexError):
                    checkfail = True
            if c and not checkfail:
                er_inner = [e + (rp.FILAMENT_WF - max(c[0],
                                 c[1] + (c[2]/np.sqrt(e + c[3]
                                                      + rp.FILAMENT_WF))))
                            for e in er]  # energies at which scattering occurs
        else:
            try:
                v0r = float(rp.V0_REAL)
            except (ValueError, TypeError):
                checkfail = True
            else:
                er_inner = [e + v0r for e in er]
        if not checkfail:
            if (psmin > min(er_inner) or psmax < max(er_inner)):
                if (psmin > min(er_inner) and psmin <= 20.
                        and psmax >= max(er_inner)):
                    # can lead to re-calculation of phaseshifts every run if
                    #  V0r as calculated by EEASiSSS differs from 'real' V0r.
                    #  Don't automatically correct.
                    logger.warning(
                        "Lowest value in PHASESHIFTS file ({:.1f} "
                        "eV) is larger than the lowest predicted scattering "
                        "energy ({:.1f} eV). If this causes problems in the "
                        "reference calculation, try deleting the PHASESHIFTS "
                        "file to generate a new one, or increase the starting "
                        "energy in the THEO_ENERGIES parameter."
                        .format(psmin, min(er_inner)))
                else:
                    logger.warning(
                        "The energy range found in the PHASESHIFTS"
                        " file is smaller than the energy range requested for "
                        "theoretical beams. A new PHASESHIFTS file will be "
                        "generated.")
                    newpsGen, newpsWrite = True, True
        else:
            logger.warning(
                "Could not check energy range in PHASESHIFTS "
                "file. If energy range is insufficient, try deleting the "
                "PHASESHIFTS file to generate a new one.")
    return (firstline, phaseshifts, newpsGen, newpsWrite)


def ps_list_to_array(ps_list):
    n_energies = len(ps_list)
    ps_energy_values = np.array([ps_list[ii][0] for ii in range(n_energies)])

    n_species = len(ps_list[0][1])
    l_max = len(ps_list[0][1][0])

    phaseshifts = np.full(
        shape=(n_species, n_energies, l_max), fill_value=np.nan)
    for en in range(n_energies):
        for elem_id in range(n_species):
            phaseshifts[elem_id, en, :] = ps_list[en][1][elem_id]

    return ps_energy_values, phaseshifts

# could easily be vectorized


def interpolate_phaseshift(phaseshifts, ps_energies, interp_energy, el_id, l):
    return np.interp(interp_energy, ps_energies, phaseshifts[el_id, :, l])


def regrid_phaseshifts(old_grid, new_grid, phaseshifts):
    n_elem, n_en, n_l = phaseshifts.shape
    new_phaseshifts = np.full(
        shape=(n_elem, len(new_grid), n_l), fill_value=np.nan)
    for l in range(n_l):
        for el in range(n_elem):
            for en_id in range(len(new_grid)):
                new_phaseshifts[el, en_id, l] = interpolate_phaseshift(
                    phaseshifts, old_grid, en_id, el, l)
    return new_phaseshifts


# TODO: We should consider a spline interpolation instead of a linear
def interpolate_phaseshifts(phaseshifts, l_max, energies):
    """Interpolate phaseshifts for a given site and energy.
    """
    stored_phaseshift_energies = [entry[0] for entry in phaseshifts]
    stored_phaseshift_energies = np.array(stored_phaseshift_energies)

    stored_phaseshifts = [entry[1] for entry in phaseshifts]
    # covert to numpy array, indexed as [energy][site][l]
    stored_phaseshifts = np.array(stored_phaseshifts)

    if (min(energies) < min(stored_phaseshift_energies)
        or max(energies) > max(stored_phaseshift_energies)):
        raise ValueError("Requested energies are out of range the range for the"
                         "loaded phaseshifts.")

    n_sites = stored_phaseshifts.shape[1]
    # interpolate over energies for each l and site
    interpolated = np.empty(shape=(len(energies), n_sites, l_max + 1),
                            dtype=np.float64)

    for l in range(l_max + 1):
        for site in range(n_sites):
            interpolated[:, site, l] = np.interp(energies,
                                                stored_phaseshift_energies,
                                                stored_phaseshifts[:, site, l])
    return interpolated
