#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SmoothClassNLLCriterion.c"
#else


void THNN_(SmoothClassNLLCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight)


{
  int n_dims = THTensor_(nDimension)(input);
  int n_classes = THTensor_(size)(input, n_dims - 1);

  if (THTensor_(nDimension)(target) > 2) {
    THError("target tensor should be 1D or 2D");
  }
  if (THTensor_(nDimension)(input) > 2) {
    THError("input tensor should be 1D or 2D");
  }

  input = THTensor_(newContiguous)(input);
  target = THTensor_(newContiguous)(target);
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;

  real *input_data = THTensor_(data)(input);
  real *target_data = THTensor_(data)(target);
  real *weights_data = weights ? THTensor_(data)(weights) : NULL;
  real *output_data = THTensor_(data)(output);
  real *total_weight_data = THTensor_(data)(total_weight);

  output_data[0] = total_weight_data[0] = 0.0;

  if (THTensor_(nDimension)(input) == 1) {
    int j;
    int dim_size = THTensor_(size)(input, 0);
    for (j = 0; j < dim_size; j++) {
      int cur_target = target_data[j];
      THAssert(cur_target >=0 && cur_target <= 1);

      real cur_weight = weights ? weights_data[j] : 1.0f;
      total_weight_data[0] += cur_weight;
      output_data[0] -= cur_target * input_data[j] * cur_weight;
    } else if (THTensor_(nDimension)(input) == 2) {
    int batch_size = THTensor_(size)(input, 0);
    int dim_size = THTensor_(size)(input, 1);
    THAssert(THTensor_(size)(target, 0) == batch_size);
    THAssert(THTensor_(size)(target, 1) == dim_size);

    int i;
    for (i = 0; i < batch_size; i++) {
      int k;
      for (k = 0; k < dim_size; k ++) {
          int cur_target = target_data[i * dim_size + k];
          THAssert(cur_target >=0 && cur_target <= 1);

          real cur_weight = weights ? weights_data[k] : 1.0f;
          total_weight_data[0] += cur_weight;
          output_data[0] -= input_data[i * dim_size + k] * cur_target * cur_weight;
      }
    }
   }
  }


  if (sizeAverage && total_weight_data[0]) {
    output_data[0] /= total_weight_data[0];
  }

  if (weights) {
    THTensor_(free)(weights);
  }
  THTensor_(free)(input);
  THIndexTensor_(free)(target);
}


void THNN_(ClassNLLCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight)
{
  int n_dims = THTensor_(nDimension)(input);
  int n_classes = THTensor_(size)(input, n_dims - 1);

  if (!THTensor_(isContiguous)(gradInput)) {
    THError("gradInput must be contiguous");
  }

  real *total_weight_data = THTensor_(data)(total_weight);

  if (!(*total_weight_data > 0)) {
    return;
  }

  if (THTensor_(nDimension)(target) > 2) {
    THError("target tensor should be 1D or 2D");
  }
  if (THTensor_(nDimension)(input) > 2) {
    THError("input tensor should be 1D or 2D");
  }

  target = THTensor_(newContiguous)(target);
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;

  real *target_data = THIndexTensor_(data)(target);
  real *weights_data = weights ? THTensor_(data)(weights) : NULL;
  real *gradInput_data = THTensor_(data)(gradInput);

  if (THTensor_(nDimension)(input) == 1) {
      int j;
      int dim_size = THTensor_(size)(input, 0);
      for (j = 0; j < dim_size; j++) {
         int cur_target = target_data[j];
         THAssert(cur_target >= 0 && cur_target <= 1);
         gradInput_data[j] =
           (!sizeAverage && weights) ? -weights_data[j] * target_data[j] : -target_data[j];
      }
  } else if (THTensor_(nDimension)(input) == 2) {
   int batch_size = THTensor_(size)(input, 0);
   int dim_size = THTensor_(size)(input, 1);
   THAssert(THTensor_(size)(target, 0) == batch_size && THAssert_(size)(target, 1) == dim_size);

   int i;
   for (i = 0; i < batch_size; i++) {
    int k;
    for (k = 0; k < batch_size; k++) {
      int cur_target = target_data[i * dim_size + j];

      THAssert(cur_target >= 0 && cur_target <= 1);

      gradInput_data[i * dim_size + k] = -(weights ? weights_data[k] * target_data[k] : target_data[k]);

      if (sizeAverage && *total_weight_data) {
       gradInput_data[i * dim_size + k] /= *total_weight_data;
      }
    }
   }
  }

  THIndexTensor_(free)(target);
  if (weights) {
    THTensor_(free)(weights);
  }
}

#endif
