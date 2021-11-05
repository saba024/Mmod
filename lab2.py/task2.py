from math import factorial, inf
import simpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig, axs = plt.subplots(2)


class QueuingSystemqueue_system_model:
    def __init__(self, env, channels_number, serv_flow, app_flow, queue_flow=None,
                 max_queue_length=inf):
        self.env = env
        self.serv_flow = serv_flow
        self.app_flow = app_flow
        self.queue_flow = queue_flow
        self.max_queue_length = max_queue_length
        self.wait_times = []
        self.qs_list = []
        self.queue_times = []
        self.queue_list = []
        self.app_requested = []
        self.app_rejected = []
        self.channel = simpy.Resource(env, channels_number)

    def application_processing(self, application):
        yield self.env.timeout(np.random.exponential(1 / self.serv_flow))

    def application_waiting(self, application):
        yield self.env.timeout(np.random.exponential(1 / self.queue_flow))


def send_application(env, application, queue_system_model):
    queue_applications_amount = len(queue_system_model.channel.queue)
    processing_applications_amount = queue_system_model.channel.count

    queue_system_model.qs_list.append(queue_applications_amount + processing_applications_amount)
    queue_system_model.queue_list.append(queue_applications_amount)

    with queue_system_model.channel.request() as request:
        current_queue_len = len(queue_system_model.channel.queue)
        current_count_len = queue_system_model.channel.count
        if queue_system_model.max_queue_length != inf:
            if current_queue_len <= queue_system_model.max_queue_length:
                start_time = env.now
                queue_system_model.app_requested.append(current_queue_len + current_count_len)
                # An application ether waits for a channel to become free or starts to process
                res = yield request | env.process(queue_system_model.application_waiting(application))
                queue_system_model.queue_times.append(env.now - start_time)
                if request in res:
                    yield env.process(queue_system_model.application_processing(application))
                queue_system_model.wait_times.append(env.now - start_time)
            else:
                queue_system_model.app_rejected.append(channels_number + max_queue_length + 1)
                queue_system_model.queue_times.append(0)
                queue_system_model.wait_times.append(0)
        else:
            start_time = env.now
            queue_system_model.app_requested.append(current_queue_len + current_count_len)
            yield request
            queue_system_model.queue_times.append(env.now - start_time)
            yield env.process(queue_system_model.application_processing(application))
            queue_system_model.wait_times.append(env.now - start_time)


def run_queue_system_model(env, queue_system_model):
    application = 0

    while True:
        yield env.timeout(np.random.exponential(1 / queue_system_model.app_flow))
        env.process(send_application(env, application, queue_system_model))
        application += 1


def find_average_qs_len(qs_list):
    average_qs_len = np.array(qs_list).mean()
    print('Average amount of applications in QS (both processing and waiting): ' + str(average_qs_len))
    return average_qs_len


def find_average_wait_time(wait_times):
    average_wait_time = np.array(wait_times).mean()
    print('Average time in QS is: ' + str(average_wait_time))
    return average_wait_time


def find_average_queue_len(queue_list):
    average_queue_len = np.array(queue_list).mean()
    print('Average queue length is: ' + str(average_queue_len))
    return average_queue_len


def find_average_queue_time(queue_times):
    average_queue_time = np.array(queue_times).mean()
    print('Average time in queue is: ' + str(average_queue_time))
    return average_queue_time


def find_empiric_probabilities(app_requested, app_rejected, queue_times, wait_times, qs_list,
                               queue_list, num_channel,
                               max_queue_length,
                               app_flow,
                               serv_flow):
    print('-------------------Empiric---------------------')
    total_applications_amount = len(app_requested) + len(app_rejected)
    P = []
    for value in range(1, num_channel + max_queue_length + 1):
        P.append(len(app_requested[app_requested == value]) / total_applications_amount)
    print('Empiric final probabilities:')
    for index, p in enumerate(P):
        print('P' + str(index) + ': ' + str(p))
    P_reject = len(app_rejected) / total_applications_amount
    print('Empiric probability of rejection: ' + str(P_reject))
    Q = 1 - P_reject
    print('Empiric Q: ' + str(Q))
    A = app_flow * Q
    print('Empiric A: ' + str(A))
    find_average_queue_len(queue_list)
    find_average_qs_len(qs_list)
    find_average_queue_time(queue_times)
    average_full_channels = Q * app_flow / serv_flow
    print('Average amount of busy channels: ' + str(average_full_channels))
    find_average_wait_time(wait_times)
    axs[0].hist(wait_times, 50)
    axs[0].set_title('Wait times')
    axs[1].hist(qs_list, 50)


def find_theoretical_probabilities(num_channel, max_queue_length, app_flow, serv_flow,
                                   queue_flow):
    print('-------------------Theoretical---------------------')
    ro = app_flow / serv_flow
    betta = queue_flow / serv_flow
    print('Theoretical final probabilities:')
    P = 0
    p0 = (sum([ro ** i / factorial(i) for i in range(num_channel + 1)]) +
          (ro ** num_channel / factorial(num_channel)) *
          sum([ro ** i / (np.prod([num_channel + t * betta for t in range(1, i + 1)])) for i in
               range(1, max_queue_length + 1)])) ** -1
    print('P0: ' + str(p0))
    P += p0
    for i in range(1, num_channel + 1):
        px = (ro ** i / factorial(i)) * p0
        P += px
        print('P' + str(i) + ': ' + str(px))
    pn = px
    pq = px
    for i in range(1, max_queue_length):
        px = (ro ** i / np.prod([num_channel + t * betta for t in range(1, i + 1)])) * pn
        P += px
        if i < max_queue_length:
            pq += px
        print('P' + str(num_channel + i) + ': ' + str(px))
    P = px
    print('Theoretical probability of rejection: ' + str(P))
    Q = 1 - P
    print('Theoretical Q: ', Q)
    A = Q * app_flow
    print('Theoretical A: ', A)
    L_q = sum([i * pn * (ro ** i) / np.prod([num_channel + t * betta for t in range(1, i + 1)]) for
               i in range(1, max_queue_length + 1)])
    print('Average queue length is: ', L_q)
    L_pr = sum([index * p0 * (ro ** index) / factorial(index) for index in range(1, num_channel + 1)]) + sum(
        [(num_channel + index) * pn * ro ** index / np.prod(
            np.array([num_channel + t * betta for t in range(1, index + 1)])) for
         index in range(1, max_queue_length + 1)])
    print('Average amount of applications in QS is: ', L_pr)
    print('Average time in queue is: ', Q * ro / app_flow)
    print('Average amount of busy channels: ', Q * ro)
    print('Average time in QS is: ', L_pr / app_flow)


if __name__ == '__main__':
    print('First example:')
    channels_number = 2
    print('Amount of channels (n): ' + str(channels_number))
    serv_flow = 4
    print('Service flow rate (mu): ' + str(serv_flow))
    app_flow = 3
    print('Applications flow rate (lambda): ' + str(app_flow))
    queue_flow = 1
    print('Queue waiting flow rate (v): ' + str(queue_flow))
    max_queue_length = 2
    print('Max queue length (m): ' + str(max_queue_length))

    env = simpy.Environment()
    queue_system_model = QueuingSystemqueue_system_model(env, channels_number, serv_flow, app_flow, queue_flow,
                               max_queue_length)
    print('Running simulation...')
    env.process(run_queue_system_model(env, queue_system_model))
    env.run(until=100)

    find_empiric_probabilities(np.array(queue_system_model.app_requested), np.array(queue_system_model.app_rejected),
                               np.array(queue_system_model.queue_times),
                               np.array(queue_system_model.wait_times),
                               np.array(queue_system_model.qs_list),
                               np.array(queue_system_model.queue_list), channels_number, max_queue_length, app_flow,
                               serv_flow)
    find_theoretical_probabilities(channels_number, max_queue_length, app_flow, serv_flow,
                                   queue_flow)
    print('End of first example')
    plt.show()


    