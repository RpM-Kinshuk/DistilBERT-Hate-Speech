import numpy as np
import logging

logger = logging.getLogger(__name__)
UNKNOWN = 'unknown'

class CHANNELS():
    UNKNOWN = UNKNOWN
    FIRST = 'first'
    LAST = 'last'

def channel_str(channel):
        if channel == CHANNELS.FIRST:
            return "FIRST"
        elif channel == CHANNELS.LAST:
            return "LAST"
        else:
            return "UNKNOWN"

def conv2D_Wmats(Wtensor, channels=CHANNELS.LAST):
        """Extract W slices from a 4 layer_id conv2D tensor of shape: (N,M,i,j) or (M,N,i,j).  
        Return ij (N x M) matrices, with receptive field size (rf) and channels flag (first or last)"""
        
        logger.debug("conv2D_Wmats")
        
        # TODO:  detect or use CHANNELS
        # if channels specified ...
    
        Wmats = []
        s = Wtensor.shape
        N, M, imax, jmax = s[0], s[1], s[2], s[3]
        
        if N + M >= imax + jmax:
            detected_channels= CHANNELS.LAST
        else:
            detected_channels= CHANNELS.FIRST
            

        if channels == CHANNELS.UNKNOWN :
            logger.debug(f"channels UNKNOWN, detected {channel_str(detected_channels)}")
            channels= detected_channels

        if detected_channels == channels:
            if channels == CHANNELS.LAST:
                logger.debug("channels Last tensor shape: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))                
                for i in range(imax):
                    for j in range(jmax):
                        W = Wtensor[:, :, i, j]
                        if W.shape[0] < W.shape[1]:
                            N, M = M, N
                            W = W.T
                        Wmats.append(W)
                        
            else: #channels == CHANNELS.FIRST  # i, j, M, N
                M, N, imax, jmax = imax, jmax, N, M
                # check this       
                logger.debug("channels First shape: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))                
                for i in range(imax):
                    for j in range(jmax):
                        W = Wtensor[i, j, :, :]
                        if W.shape[1] < W.shape[0]:
                            N, M = M, N
                            W = W.T
                        Wmats.append(W)
                            
        elif detected_channels != channels:
            logger.warning("warning, expected channels {},  detected channels {}".format(channel_str(channels),channel_str(detected_channels)))
            # flip how we extract the Wmats
            # reverse of above extraction
            if detected_channels == CHANNELS.LAST:
                logger.debug("Flipping LAST to FIRST Channel, {}x{} ()x{}".format(N, M, imax, jmax))   
                for i in range(N):
                    for j in range(M):
                        W = Wtensor[i, j,:,:]
                        if imax < jmax:
                            W = W.T
                        Wmats.append(W)
                        
            else: #detected_channels == CHANNELS.FIRST:
                N, M, imax, jmax = imax, jmax, N, M   
                logger.debug("Flipping FIRST to LAST Channel, {}x{} ()x{}".format(N, M, imax, jmax))                
                # check this       
                for i in range(N):
                    for j in range(M):
                        W = Wtensor[:, :, i, j]
                        if imax < jmax:
                            W = W.T
                        Wmats.append(W)
            # final flip            
            N, M, imax, jmax = imax, jmax, N, M   
           
                
        rf = imax * jmax  # receptive field size             
        logger.debug("get_conv2D_Wmats N={} M={} rf= {} channels= {}".format(N, M, rf, channels))
    
        return Wmats, N, M, rf