"""
Extra utility functions for doing odd jobs
"""

import numpy as np

def paramsumstr(iterable):
    output = '|'
    for x in iterable:
        output += x+'|'
    return output

def recursing_combinations(combfunc,lol):
    #lol is list of lists
    #combfunc is a handle to a function that takes an iterable
    #and specifies how to combine them
    outlist = []
    def recurse(positions):
        #positions is a list of indexes giving positions within each list of lol
        #positions is of the same length as lol (number of lists in the list)
        #should always begin with just zeroes
        #print(positions)
        outlist.append(combfunc([lol[x][positions[x]] for x in range(len(positions))]))
        lastelementcheck = True
        for x in range(len(positions)):
            if (positions[x] < len(lol[x])-1):
                lastelementcheck = False
        if lastelementcheck:
            return
        for dex in range(len(positions)-1,-1,-1):
            if positions[dex] < (len(lol[dex])-1):
                #if not at the end of a row, move along the row by one and recurse
                positions[dex] += 1
                recurse(positions)
                return
            elif dex > 0:
                #reset row, ++previous row
                #keep looping to carry as many times as needed
                positions[dex] = 0
                n=1
                while True:
                    if dex-n < 0:
                        break
                    elif positions[dex-n] < (len(lol[dex-n])-1):
                        positions[dex-n] += 1
                        recurse(positions)
                        return
                    else:
                        positions[dex-n] = 0
                        n+=1
                    
    initialpositions = []
    for x in lol:
        initialpositions.append(0)
    recurse(initialpositions)
    return outlist


def interp_find_x_intercept(xdat,ydat):
    # for xdata and ydata, find the x-intercept, use simple interp
    # assumes a simple monotonic gradient
    # easiest case, if exact x-intercept exists, just return it

    # convert to lists, if np.ndarrays, can't use .index()
    xdat = list(xdat)
    ydat = list(ydat)
    try: 
        return xdat[ydat.index(0)]
    except:
        deltay = [-np.inf,np.inf] # current closest yvals to zero
        deltax = [0,0]
        for yval in ydat:
            if 0 > yval > deltay[0]:
                deltay[0] = yval
                deltax[0] = xdat[ydat.index(yval)]
            if 0 < yval < deltay[1]:
                deltay[1] = yval
                deltax[1] = xdat[ydat.index(yval)]
        m = (deltay[1]-deltay[0])/(deltax[1]-deltax[0])
        c = deltay[1] - m*deltax[1]
        return -c/m

def simple_dec(dat: np.ndarray, targlen: int) -> np.ndarray:
    # simple decimator, no filtering
    # expects dat to be pairs of columns, chops data aiming to make
    # length as close to targlen as possible.
    # if current length is longer than targlen, just return the old array
    nl = dat.dtype.names
    oldlen = dat.shape[0]
    if oldlen < targlen:
        return dat
    rate = int(np.floor(oldlen/targlen))
    newlen = int(np.floor(dat.shape[0]/rate))
    outarr = np.ndarray(shape=(newlen,), dtype=dat.dtype)
    for n in nl:
        for a in range(oldlen):
            if not(a%rate) and int(a/rate) < newlen:
                outarr[n][int(a/rate)] = dat[n][a]
    return outarr

def simplediff(linx,liny):
    # simple numerical differentiation of data arrays
    # this method gives output vectors that are the same length
    # as the input vectors, the cost is that the first and
    # last values will have a little more error than the
    # ones in the middle of the array.
    louty = []
    if len(linx) != len(liny):
        print('Error: input vectors different length!')
        return liny
    for a in range(len(liny)):
        if a == 0:
            tang = (liny[a+1]-liny[a])/(linx[a+1]-linx[a])
        elif a == (len(liny)-1):
            tang = (liny[a]-liny[a-1])/(linx[a]-linx[a-1])
        else:
            y2 = (liny[a+1]-liny[a])/2
            y1 = (liny[a]-liny[a-1])/2
            x2 = (linx[a+1]-linx[a])/2
            x1 = (linx[a]-linx[a-1])/2
            tang = (y2+y1)/(x2+x1)
        louty.append(tang)
    return louty
