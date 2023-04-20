# [APP] Image Classification using Convolution Neural Network
List of members: Mai Hồng Phúc - 19120620
                 Hoàng Anh Quân -19120628
                 Nguyễn Anh Quốc - 19120633
                 
Keywords:  CNN, parallel programming, image classification, CUDA, Deep learning

List of references: materials that have been used in this research
https://www.linkedin.com/pulse/forward-back-propagation-over-cnn-code-from-scratch-coy-ulloa/
https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
https://arxiv.org/pdf/1511.08458.pdf
https://jmyao17.github.io/Machine_Learning/Neural_Network/CNN-1/CNN_Build.html

Content
Summary:
Dự án của nhóm sẽ thực hiện song song hóa ứng dụng phân lớp hình ảnh sử dụng mạng Convolution Neural Network (CNN). Mạng CNN gồm nhiều lớp kết hợp lại với nhau: lớp convolution, lớp pooling, mạng dense,... Nhóm sẽ thực hiện song song hóa các bước thành phần và cả quá trình back propagation với hy vọng tốc độ thực thi của ứng dụng sẽ được tăng nhanh hơn so với phiên bản tuần tự của nó. Ứng dụng sẽ được thực hiện trên NVIDIA CUDA GPU.

Background
Trong những năm trở lại đây đã có sự nghiên cứu phát triển vượt bậc về việc sử dụng CNN cho các nhiệm vụ liên quan đến nhận dạng và phân loại hình ảnh. Ý tưởng chính của mô hình CNN là lấy hình ảnh làm đầu vào và rút trích các đặc trưng phức tạp hơn ở mỗi lớp ở độ phân giải thấp hơn. Sau các lớp rút trích này, có một vài lớp kích hoạt để đưa ra quyết định, dự đoán về hình ảnh.
Việc đào tạo mô hình CNN cần một lượng lớn dữ liệu để hội tụ về cực tiểu toàn cục của chúng. Hệ quả là, việc đào tạo này rất nặng về tính toán và có thể mất nhiều giờ, thậm chí nhiều ngày để chạy trên các CPU truyền thống.
Trong dự án này, nhóm mong muốn khai thác tính song song trong từng lớp của CNN như Pooling, FullyConnected, Convolution,..và sửa đổi thuật toán lan truyền ngược để phù hợp với kiến trúc CUDA và đạt được tốc độ cao.

The challenge
Việc đào tạo mạng nơ-ron tích chập (CNN) là một nhiệm vụ tính toán chuyên sâu nên yêu cầu đòi hỏi khả năng song song hóa hiệu quả để rút ngắn thời gian thực hiện.  Dữ liệu training cho mang nơ-ron ngày càng lớn nên việc song song hóa việc đào tạo mạng CNN trở nên quan trọng hơn. Các vấn đề, khó khăn gặp phải khi thực hiện song song hóa là:
Kết quả tính toán của mỗi lớp cần được chia sẻ giữa tất cả các thread vì tính toán của lớp tiếp theo phụ thuộc vào kết quả của lớp trước. Do đó, CNN có tỷ lệ giao tiếp để tính toán rất cao dẫn đến cường độ thực hiện song song trên GPU thấp
Giới hạn bộ nhớ: Số lượng mẫu đào tạo cho việc tính toán song song bị hạn chế đáng kể bởi vì bộ nhớ chung có sẵn cho GPU thấp.
Các lớp phụ thuộc vào nhau nên chỉ có sự song song trong một lớp chứ không phải giữa các lớp
Để vượt qua những thử thách kể trên, yêu cầu nhóm tối ưu hóa kiến trúc CNN để sử dụng ít bộ nhớ hơn và ít yêu cầu giao tiếp hơn, sử dụng lại bộ nhớ để không phải tải nhiều lần từ CPU sang GPU.

Resources
Đối với đề tài này nhóm sử dụng máy chủ Colab (Tên CPU: Persitence M, 13GB Ram, Tên GPU: Tesla T4) để tiến hành cài đặt chương trình tuần tự và song song cho việc huấn luyện mô hình CNN. Mã nguồn cài đặt mô hình CNN sẽ xây dựng từ đầu, có thể sử dụng một số thư viện hỗ trợ như numpy, thư viện được sử dụng để cài đặt song song là Numba. Bài báo mô tả về mô hình CNN: https://arxiv.org/pdf/1511.08458.pdf. Các tài nguyên về dữ liệu huấn luyện phân loại ảnh đều đã có sẵn trên các thư viện của tensorflow như MNIST, CIFAR-10,...

Goals And Deliverables
Trong dự án này, nhóm sẽ thực hiện song song hóa mạng CNN bắt đầu từ phiên bản tuần tự. Mục tiêu mà nhóm đặt ra là xây dựng được phiên bản song song với tốc độ thực thi tương đương (hoặc chậm hơn không quá nhiều) so với mô hình được xây dựng từ các thư viện chuẩn (keras, tensorflow, pytorch,...) vì đây là các framework được tối ưu về tốc độ thực thi. Hoặc ít nhất là ứng dụng được tạo ra chạy nhanh hơn so với phiên bản tuần tự. Nếu tiến độ của dự án nhanh hơn và tốt hơn, nhóm có thể sẽ tối ưu ứng dụng về dung lượng bộ nhớ, hoặc song song hóa các lớp thành phần trên tập dữ liệu 3D.
Nhóm sẽ demo ứng dụng song song của mình đang hoạt động và so sánh thời gian chạy trên các tập dữ liệu khác nhau. Nhóm cũng sẽ trình bày các biểu đồ tốc độ thực thi so sánh các phiên bản tuần tự và song song của mạng CNN cũng như mô tả một số cách tối ưu hóa mà nhóm đã sử dụng.
Các câu hỏi mà nhóm cần trả lời trong phân tích dự án: Để thực hiện dự án thì cần làm những công việc gì và cần thời gian bao lâu? Công việc của mỗi thành viên được phân chia như thế nào? Thời gian thực hiện cho từng công việc là bao lâu? Nếu có một thành viên chưa hoàn thành xong công việc trước thời hạn thì nên giải quyết thế nào? Những rủi ro nào có thể xảy ra trong quá trình thực hiện dự án? Liệu dự án có cho ra được kết quả như ý muốn? Việc tối ưu thời gian của ứng dụng có phải là điều cần thiết? Lợi ích lớn nhất của dự án là gì?
Ứng dụng có thể song song các tác vụ độc lập như phép tích chập, nhân ma trận, tính toán trên từng batch,... Hiệu suất của phiên bản song song mà nhóm hy vọng đạt được có thể nhanh gấp khoảng 10-30 lần so với việc cài đặt tuần tự.
