# import torch
# from torch.autograd import Variable
# import torch.nn.functional as F
# import time
# import os
# import sys
# import json
# import pdb

# from libs.utils import AverageMeter

import torch
import torch.nn.functional as F
import time
import os
import json
from tqdm import tqdm
from libs.utils import AverageMeter

def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=10)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    test_results['results'][video_id] = video_results

def test(data_loader, model, opt, class_names):
    print('test')

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            
            inputs = inputs.to(device)

            if opt.is_place_adv and opt.is_mask_adv:
                outputs, _, _ = model(inputs)
            elif opt.is_place_adv:
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)

            if not opt.no_softmax_in_test:
                outputs = F.softmax(outputs, dim=1)

            for j in range(outputs.size(0)):
                current_video_id = targets[j]
                if not (i == 0 and j == 0) and current_video_id != previous_video_id:
                    calculate_video_results(output_buffer, previous_video_id, test_results, class_names)
                    output_buffer = []

                output_buffer.append(outputs[j].detach().cpu())
                previous_video_id = current_video_id

            if (i % 100) == 0:
                with open(os.path.join(opt.result_path, f'{opt.test_subset}.json'), 'w') as f:
                    json.dump(test_results, f)

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\tTime {:.3f} ({:.3f})\tData {:.3f} ({:.3f})'.format(
                i + 1, len(data_loader),
                batch_time.val, batch_time.avg,
                data_time.val, data_time.avg))

    with open(os.path.join(opt.result_path, f'{opt.test_subset}.json'), 'w') as f:
        json.dump(test_results, f)



def test_scene_accuracy(data_loader, model, opt):
    import time
    from torch.autograd import Variable
    from libs.utils import AverageMeter, calculate_accuracy_pt_0_4

    model.eval()
    batch_time = AverageMeter()
    place_accuracies = AverageMeter()

    torch_version = float(torch.__version__[:3])
    end_time = time.time()

    for i, (inputs, targets, places) in tqdm(enumerate(data_loader)):
        places = places.cuda(non_blocking=True)
        if opt.model == 'vgg':
            inputs = inputs.squeeze()

        with torch.no_grad():
            inputs = Variable(inputs)

            # Get place output
            if opt.is_place_adv:
                _, outputs_place = model(inputs)
            else:
                print("ERROR: Model was not trained with is_place_adv=True.")
                return

            # Compute accuracy
            if torch_version < 0.4:
                place_acc = calculate_accuracy(outputs_place, places)
            else:
                place_acc = calculate_accuracy_pt_0_4(outputs_place, places)

            place_accuracies.update(place_acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Test Batch: [{}/{}]\tTime {:.3f}\tScene Acc {:.3f} ({:.3f})'.format(
            i + 1, len(data_loader), batch_time.val, place_accuracies.val, place_accuracies.avg))

    print('===> Final Scene (Place) Classification Accuracy: {:.3f}%'.format(place_accuracies.avg * 100))
    return place_accuracies.avg

# # PyTorch 0.3
# def calculate_video_results(output_buffer, video_id, test_results, class_names):
#     video_outputs = torch.stack(output_buffer)
#     average_scores = torch.mean(video_outputs, dim=0)
#     sorted_scores, locs = torch.topk(average_scores, k=10)

#     video_results = []
#     for i in range(sorted_scores.size(0)):
#         video_results.append({
#             'label': class_names[locs[i]],
#             'score': sorted_scores[i]
#         })

#     test_results['results'][video_id] = video_results

# # PyTorch 0.4
# def calculate_video_results_pt_0_4(output_buffer, video_id, test_results, class_names):
#     video_outputs = torch.stack(output_buffer)
#     average_scores = torch.mean(video_outputs, dim=0)
#     sorted_scores, locs = torch.topk(average_scores, k=10)

#     video_results = []
#     for i in range(sorted_scores.size(0)):
#         video_results.append({
#             'label': class_names[locs[i].item()],
#             'score': sorted_scores[i].cpu().numpy().item()
#         })

#     test_results['results'][video_id] = video_results
    

# def test(data_loader, model, opt, class_names):
#     print('test')

#     model.eval()

#     # pytroch version check
#     torch_version = float(torch.__version__[:3])

#     batch_time = AverageMeter()
#     data_time = AverageMeter()

#     end_time = time.time()
#     output_buffer = []
#     previous_video_id = ''
#     test_results = {'results': {}}
#     for i, (inputs, targets) in enumerate(data_loader):
#         data_time.update(time.time() - end_time)

#         inputs = Variable(inputs, volatile=True)
#         outputs = model(inputs)
#         if not opt.no_softmax_in_test:
#             outputs = F.softmax(outputs)

#         for j in range(outputs.size(0)):
#             if not (i == 0 and j == 0) and targets[j] != previous_video_id:
#                 if torch_version < 0.4:
#                     calculate_video_results(output_buffer, previous_video_id,
#                                             test_results, class_names)
#                 else:
#                     calculate_video_results_pt_0_4(output_buffer, previous_video_id,
#                                             test_results, class_names)
#                 output_buffer = []
#             output_buffer.append(outputs[j].data.cpu())
#             previous_video_id = targets[j]

#         if (i % 100) == 0:
#             with open(
#                     os.path.join(opt.result_path, '{}.json'.format(
#                         opt.test_subset)), 'w') as f:
#                 json.dump(test_results, f)

#         batch_time.update(time.time() - end_time)
#         end_time = time.time()

#         print('[{}/{}]\t'
#               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#               'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
#                   i + 1,
#                   len(data_loader),
#                   batch_time=batch_time,
#                   data_time=data_time))
#     with open(
#             os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),
#             'w') as f:
#         json.dump(test_results, f)
