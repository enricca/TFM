ó
wíûZc        	   @   s'  d  Z  d d l j j Z d d l Z d d l Z d d l Z d d l	 Z	 d d l
 j Z d d l Z d d l Z d d l Z d   Z d d d g d  Z e j d d d g d d d g d d d g d d d g d d d g d d d g g  d  Z d d	  Z d d
  Z d   Z d d  Z d S(   s=   
File for reading the manual segmentation done by clinicians
iÿÿÿÿNc            su  i  } x/t    f d   t j     D]} t j j   |  } xî t  d   t j |   D]Ñ } t j t j j | |   } | j   } | j   d j	 j
 d  d d }	 y& t | j d |	 |	 f  j  }
 Wn) t | j d |	 |	 f  j  }
 n X|
 | k rq_ n  t j j | |  | |
 <q_ Wq( Wt j j i | j   d 6| j   d 6 } | j |  d	 S(
   sJ   
    Generates a path that links series instanceUID with the xml file
    c            s/   |  j  d  o. t j j t j j   |    S(   Nt   .(   t
   startswitht   ost   patht   isdirt   join(   t   s(   t   xmlPath(    s   readingXMLSegmentation.pyt   <lambda>   s    c         S   s   |  j  d  o |  j d  S(   NR    s   .xml(   R   t   endswith(   R   (    (    s   readingXMLSegmentation.pyR      s    i    t   }s$   %sResponseHeader/%sSeriesInstanceUids&   %sResponseHeader/%sCTSeriesInstanceUidt   pIdR   N(   t   filterR   t   listdirR   R   t   ETt   parset   getroott   getchildrent   tagt   splitt   strt   findt   textt   pandast	   DataFramet	   from_dictt   keyst   valuest   to_csv(   R   t   patientst   outputCsvPatht   pathst	   subfolderR   t   pt   treet   roott   nDomaint	   patientIdt   pd(    (   R   s   readingXMLSegmentation.pyt   generate_csv_xml_path   s     ("$&&!,i   c         C   sõ   t  j |  |  j t  } t  j t | d | d d  t | d | d d  t | d | d d   \ } } } | | d d | | d d | | d d |  d k } t  j | | | d | | | d | | | d f  j S(   s-  
    Creates a ball of radius R, centered in the coordinates center
    @param rad: radius of the ball, in mm
    @param center: center of the ball (slice, x, y) in pixel coordinates

    @spatialSpacing

    returns a list of coordinates (x, y, z) that are in the ball of radius r centered in 0.
    i    i   i   (   t   npt   ceilt   astypet   intt   meshgridt   xranget   stackt   T(   t   radt   centert   spatialScalingt   rt   xt   yt   zt   mask(    (    s   readingXMLSegmentation.pyt   ball%   s    A$i    c         C   sª   g  } |  g } t  j | j d t } x| | r¥ | j   } | t |  s* | t |  rc q* n  | j |  t | t |  <x | D] } | j | |  q Wq* W| S(   sT   
    gets all the points in the same  connected component that contains start. 
    t   dtype(   R(   t   zerost   shapet   boolt   popt   tuplet   appendt   True(   t   startR7   t
   directionst   pointst	   toExploret   exploredR!   t   d(    (    s   readingXMLSegmentation.pyt   label_from_point9   s    		!t    c         C   s$  t  |  j d |  j  } |  j d |  j d k } |  j d | | f  } |  j d | | f  } t j g  | D] } t  | j  ^ qw  t j g  | D] } t  | j  ^ q  } } t j j | | |  \ }	 }
 t j	 | d t j
 } d | |
 |	 f <| s| d 9} n  | | f S(	   Ns   %simageZpositions   %sinclusiont   TRUEs   %sedgeMap/%sxCoords   %sedgeMap/%syCoordR9   i   iÿÿÿÿ(   t   floatR   R   t   findallR(   t   arrayt   skimaget   drawt   polygonR:   t   int16(   t   roiR;   R$   R6   t	   inclusiont
   noduleROIXt
   noduleROIYR4   R5   t   rrt   ccR7   (    (    s   readingXMLSegmentation.pyt   noduleToMaskJ   s    Wc         C   sK   xD |  j  d |  D]/ } t | | |  \ } } | | j |  q Wd  S(   Ns   %sroi(   RK   RW   R?   (   t   nodulet   allRoisR;   R$   R3   R7   R6   (    (    s   readingXMLSegmentation.pyt   parseNoduleZ   s    c         C   sO  | j    } t j | d | d | d g d t j } t j |   } | j   } | j   d j j	 d  d d } | j
 d | | f  } t j t  } x( | D]  }	 t |	 | | d  d | q§ Wx} | j   D]o }
 t t j |
 | j   d | j   d   } t j | |
 d d d	 k | | d
 d
  d
 d
  f <qØ W| S(   sE   
    Creates the nodule mask from the xml file and the sitkImage
    i   i    i   R9   R
   s&   %sreadingSession/%sunblindedReadNoduleR$   t   axisi   N(   t   GetSizeR(   R:   RP   R   R   R   R   R   R   RK   t   collectionst   defaultdictt   listRZ   R   R+   t   roundt	   GetOrigint
   GetSpacingt   sum(   R   t	   sitkImaget   sizeR7   R"   R#   R$   t   nodulest   slicest   nt   ft   nSlice(    (    s   readingXMLSegmentation.pyt   getNoduleMaskFromXMLa   s    -$1<RX   c         C   sÕ  t  |  d k r d S|  j   \ } } } t | |   } t j | j d t j } t j |  j	    }	 t j |  j
    }
 xO| j   D]A\ } } | d } | d } | d } | d } t j | | | g  } t j | |	 |
  } t j | d | d | d g d t j } | d	 k r t | |  } | d
 k rId n t j t j | | d   } t j |  } t  |  rÍ| | | d d  d f | d d  d f | d d  d f f <qÍq q W| S(   s-   
    Creates a mask from nodule csv. If 
    i    R9   t   coordXt   coordYt   coordZt   diameter_mmi   i   i   RX   g      ð?iT   N(   t   lent   NoneR\   Rk   R(   R:   R;   RP   RL   Ra   Rb   t   iterrowst   rintRG   R`   (   t   imgR   Rf   t   channelt   heightt   widtht   num_zt
   noduleMaskR7   t   origint   spacingt   indext   rowt   node_xt   node_yt   node_zt   diamR1   RC   t   value(    (    s   readingXMLSegmentation.pyt   create_mask_from_xmlv   s0    



-2M(   t   __doc__t   xml.etree.ElementTreet   etreet   ElementTreeR   t   numpyR(   RM   t   skimage.drawR   t   matplotlib.pyplott   pyplott   pltR   R]   t	   SimpleITKt   sitkR'   R8   RL   RG   RW   RZ   Rk   R   (    (    (    s   readingXMLSegmentation.pyt   <module>   s   	]	