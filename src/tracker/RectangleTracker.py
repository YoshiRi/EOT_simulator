"""Basic functions Rectangle Tracker Utility
"""
import numpy as np


def estimate_number_of_sides(measurements,threshold=25):
    """Estimate number of measured rectangle sides

    Args:
        measurements (_type_): list of measured points coorinate
        threshold (int, optional): _description_. Defaults to 25.

    Returns:
        _type_: _description_
    """
    Nz = len(measurements)
    if Nz == 1:
        N = 1
    else:
        Z = np.array(measurements).reshape(Nz,2) # to numpy array
        Zbar = np.mean(Z,axis=0)
        Znorm = Z-Zbar
        C = Znorm.T @ Znorm
        C = C/(Nz-1)
        assert C.shape == (2,2), "Real shape is " + str(C.shape)
        eigs = sorted( np.linalg.eigvals(C) )
        
        if eigs[0] == 0: # prevent zero division
            N = 1
        elif eigs[1]/eigs[0] > threshold:
            N = 1
        else:
            N = 2

    return N


def point2line_distance(z1,z2,z3):
    """return distance between point and line

    Args:
        z1 (list): point of line edge
        z2 (list): point of line edge
        z3 (list): target point to calc distance

    Returns:
        double : distance between z3 and the line passing z1 and z2
    """
    Num = np.sqrt( (z2[0]-z1[0])**2 + (z2[1]-z1[1])**2 )
    Den = (z2[0]-z1[0])*(z1[1]-z3[1]) + (z1[0]-z3[0])*(z2[1]-z1[1])
    d = np.abs(Den/Num)
    return d

def safe_mean(sum,num):
    if num > 0:
        return sum/num
    else:
        return sum

def find_corner_index(measurements):
    """Extract corner measurements

    Args:
        measurements (list of list): List of 2D measurements 
    Returns:
        corner_index : corner index 
        divided indexes: 
    """
    Nz = len(measurements)

    dmin = 1e5*Nz
    corner_index = None

    if Nz < 3:
        return corner_index

    for n in range(1,Nz-1):
        d, d_a, d_b = 0, 0, 0
        for k in range(1,n):
            d_a += point2line_distance(measurements[0],measurements[n],measurements[k])
        for k in range(n+1,Nz-1):
            d_b += point2line_distance(measurements[n],measurements[-1],measurements[k])
        d = d_a + d_b

        if d < dmin:
            corner_index = n
            dmin = d
            d_a_mean = safe_mean(d_a,len(range(1,n)))
            d_b_mean = safe_mean(d_b,len(range(n+1,Nz-1)))
            
        if d_a_mean < d_b_mean:
            parted_measurements = [measurements[:n+1], measurements[n+1:]]
        else:
            parted_measurements = [measurements[:n], measurements[n:]]
        
    return corner_index, parted_measurements


class RectangleShapePrediction():
    def __init__(self) -> None:
        self.orientation = None
        self.center = None
        self.width = None
        self.length = None

    def normal_vector_angles(self):
        """Return Normal vector angles

        front

        +--b1---+
        |       |
        |       |
        b4  ^   b2
        |   |   |
        |       |
        |       |
        +---b3--+

        Returns:
            list of normal vector angle : [b1, b2, b3, b4]
        """
        n_vecs = [self.orientation + np.pi, self.orientation + np.pi/2, self.orientation, self.orientation - np.pi/2]
        return n_vecs
    
    def find_closest_angle(self,angle_rad, threshold=0.5):
        """find closest direction

        Args:
            angle_rad (_type_): _description_
            threshold (float, optional): direction error . Defaults to 0.5 (28.6degree)

        Returns:
            int : closest direction index
        """
        angles = self.normal_vector_angles()

        for i in range(4):
            drad = rad_distance(angles[i], angle_rad)
            if drad < threshold:
                return i
        # if no closest point 
        return None

    def get_equally_divided_coords(self,division_num,w,l):
        """divide one side

        Args:
            division_num (_type_): _description_
            w (_type_): width
            l (_type_): length

        
        division_num
        ---o-----o-----o-----o---- 
                                    ^
                                    |
                                    |
                                    |
                                    | length/2
                   center           |
                                    |
                     x              v
        <---------------------------->
                    width

        Returns:
            _type_: coordinates for divided points
        """
        divided_coords = []
        y = l/2
        x0 = - w/2
        for i in range(division_num):
            dx = (2*i + 1) / division_num/ 2 * w
            divided_coords.append([x0 + dx, y])
        return np.array(divided_coords).reshape(-1,2)

    def divide_coords(self, div_num, indx):
        """Assume lidar estimation is clock wise

        Args:
            div_num (_type_): number of measurement point in one side
            indx (_type_): side index 0 is front, 1 is right, 2 is back, 3 is left 

        Returns:
            _type_: Nx2 np array [[x1,y1],[x2, y2]...]
        """
        R90 = rot_mat_2d(np.pi/2)
        R180 = rot_mat_2d(np.pi)
        center = np.array(self.center).reshape(-1,2) 
        
        if indx == 0:
            coords_ = self.get_equally_divided_coords(div_num,self.width,self.length)
            coords = center + coords_
        elif indx == 1:
            coords_ = self.get_equally_divided_coords(div_num,self.length,self.width)
            coords = np.transpose(R90.T @ coords_.T)
        elif indx == 2:
            coords_ = self.get_equally_divided_coords(div_num,self.width,self.length)
            coords = np.transpose(R180 @ coords_.T)
        elif indx == 3:
            coords_ = self.get_equally_divided_coords(div_num,self.length,self.width)
            coords = np.transpose(R90 @ coords_.T)
        else:
            coords =  None
        return coords
