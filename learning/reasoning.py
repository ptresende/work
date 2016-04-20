import copy
import math


def compare_lists(l1, l2):
  # Maybe weigh differently the equal units and equal order measures
  w_equal_units = 1#0.3
  w_equal_order = 1#0.7

  equal_units = 0
  equal_order = 0

  l1_temp = copy.deepcopy(l1)
  l2_temp = copy.deepcopy(l2)

  for c in l1:
    if c in l2:
      l1_temp.remove(c)
      l2_temp.remove(c)
      equal_units += 1

  equal_units_norm = float(equal_units) / float(len(l1))

  l1_temp = copy.deepcopy(l1)
  l2_temp = copy.deepcopy(l2)

  # Reverse the lists and
  l1_temp.reverse()
  l2_temp.reverse()

  for i in xrange(0, len(l1_temp)):
    if l1_temp[i] == l2_temp[i]:
      equal_order += 1 * float(i) / float(len(l1_temp))

  #equal_order_norm = float(equal_order) / float(len(l1))
  equal_order_norm = float(equal_order) / (float(sum(xrange(0, len(l1_temp)))) / float(len(l1_temp)))
  #equal_order_norm = float(equal_order) * (1 / math.pow(float(len(l1_temp)), 2) * sum(xrange(0, len(l1_temp))))

  comparison_rate = (equal_units_norm + equal_order_norm) / 2
  #comparison_rate = (w_equal_units * equal_units_norm + w_equal_order * equal_order_norm) / 2

  print "Equal Units:", equal_units
  print "Equal Units Norm:", equal_units_norm, "\n"
  print "Equal Order:", equal_order
  print "Equal Order Norm:", equal_order_norm, "\n"
  print "Comparison Rate:", comparison_rate





# ############ Pretty obvious
#STM = [1,2,3,4,5]
#C_better = [1,2,3,4,5]
#C_worse = [0,0,0,0,0]


#compare_lists(STM, C_better)
#print "\n\nnow the worse\n\n"
#compare_lists(STM, C_worse)


############ Pretty obvious 2
# STM = [1,2,3,4,5,6,7,8,9,10]
# C_better = [1,2,3,4,5,6,7,8,9,10]
# C_worse = [0,0,0,0,0,0,0,0,0,0]


# compare_lists(STM, C_better)
# print "\n\nnow the worse\n\n"
# compare_lists(STM, C_worse)


############ The first one is better because of the order
#STM = [1,2,3,4]
#C_better = [1,2,9,9]
#C_worse = [9,9,3,4]

#compare_lists(STM, C_better)
#print "\n\nnow the worse\n\n"
#compare_lists(STM, C_worse)



############ The first one is better because it contains more elements of the STM
STM = [1,2,3,4,5,6,7,8,9]
C_better = [1,2,3,4,9,8,7,6,5]
C_worse = [1,2,3,4,10,10,10,10,10]

compare_lists(STM, C_better)
print "\n\nnow the worse\n\n"
compare_lists(STM, C_worse)



############ Parameters should maybe be adjusted so that the first one is better?
#STM = [1,2,3,4,5,6,7,8,9]
#C_better = [1,2,3,4,5,10,10,10,10]
#C_worse = [1,2,3,4,9,8,7,6,5]

#compare_lists(STM, C_better)
#print "\n\nnow the worse\n\n"
#compare_lists(STM, C_worse)
