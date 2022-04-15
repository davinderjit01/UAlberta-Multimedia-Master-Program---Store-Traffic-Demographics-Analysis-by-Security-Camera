Attracting  customers  to  stores  for  increasing  sales is  the  backbone  for  the  financial  growth  of  retailers.  With  therevolution  in  technology,  need  of  strategies  for  increasing  thesales  is  burgeoning.  Thus  'Store  Traffic  Demographics  Analysis by Security Camera' will be a system which counts the number of  customers  going  in  and  out  of  the  store  .


For  counting  people  going  in  and  out  of  the  store,we are   firstly   detecting   them   using   background   subtractiontechniques.We   are   taking   a   background   image   and   thencomparing  each  frame  in  the  video  with  that  image.At  the first frame,for  our  case,we  are  choosing  entry  point  of  the  storeas  our  region  of  interest  only  where  we  are  performingbackground subtraction.


For   solving   the   gender   prediction   problem,   We   used ResNet34 pre-trained  model  which  was  trained  on  ImageNet dataset   using   fastai   library   in   python   created   on   top   of PyTorch.

--------------------Running the code for People Counter--------------------
>>Use command :
python people_counter.py --videos [Location of the Videos]


--------------------Running the code for Gender Prediction--------------------

Run the file after installing fastai library.

