from PIL import Image, ImageDraw
import cv2
import mediapipe as mp

def classify_line_length(tip_position, thresholds):
    """Phân loại độ dài đường chỉ tay thành 10 mức"""
    for i in range(len(thresholds)):
        if tip_position < thresholds[i]:
            return i
    return len(thresholds)  # Mức cao nhất

def calculate_line_length(line_points):
    """Tính toán chiều dài thực tế của đường chỉ tay (tính bằng pixel)"""
    if not line_points or len(line_points) < 2:
        return 0
    
    total_length = 0
    for i in range(1, len(line_points)):
        x1, y1 = line_points[i-1]
        x2, y2 = line_points[i]
        # Tính khoảng cách Euclidean giữa 2 điểm liên tiếp
        segment_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        total_length += segment_length
    
    return round(total_length)

def measure(path_to_warped_image_mini, lines):
    heart_thres_x = [0] * 9  # 9 ngưỡng cho 10 mức
    head_thres_x = [0] * 9
    life_thres_y = [0] * 9

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image = cv2.flip(cv2.imread(path_to_warped_image_mini), 1)
        image_height, image_width, _ = image.shape

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Kiểm tra nếu không phát hiện được tay
        if not results.multi_hand_landmarks:
            im = Image.open(path_to_warped_image_mini)
            draw = ImageDraw.Draw(im)
            
            # Kiểm tra nếu lines là None hoặc rỗng
            if lines is None:
                lines = []
            
            contents = []
            
            # Vẫn cố gắng xử lý các đường nếu có, nhưng không có thông tin landmark để tính toán chính xác
            if len(lines) > 0 and lines[0] is not None and len(lines[0]) > 0:
                heart_line = lines[0]
                heart_line_points = [tuple(reversed(l[:2])) for l in heart_line]
                contents.extend(['Đường tình cảm:', 'Phát hiện đường tình cảm nhưng không thể phân tích chi tiết do không nhận dạng được bàn tay. Hãy thử chụp ảnh rõ nét hơn.'])
                draw.line(heart_line_points, fill="red", width=3)
            else:
                contents.extend(['Đường tình cảm:', 'Không tìm thấy đường tình cảm trên lòng bàn tay của bạn.'])
                
            print(">>> Đường trí tuệ kiểm tra:")
            if len(lines) > 1 and lines[1] is not None and len(lines[1]) > 0:
                head_line = lines[1]
                print(">>> Đường trí tuệ:")
                head_line_points = [tuple(reversed(l[:2])) for l in head_line]
                contents.extend(['Đường trí tuệ:', 'Phát hiện đường trí tuệ nhưng không thể phân tích chi tiết do không nhận dạng được bàn tay.'])
                draw.line(head_line_points, fill="green", width=3)
            else:
                contents.extend(['Đường trí tuệ:', 'Không tìm thấy đường trí tuệ trên lòng bàn tay của bạn.'])
                
            if len(lines) > 2 and lines[2] is not None and len(lines[2]) > 0:
                life_line = lines[2]
                life_line_points = [tuple(reversed(l[:2])) for l in life_line]
                contents.extend(['Đường sinh mệnh:', 'Phát hiện đường sinh mệnh nhưng không thể phân tích chi tiết do không nhận dạng được bàn tay.'])
                draw.line(life_line_points, fill="blue", width=3)
            else:
                contents.extend(['Đường sinh mệnh:', 'Không tìm thấy đường sinh mệnh trên lòng bàn tay của bạn.'])
            
            return im, contents
        
        hand_landmarks = results.multi_hand_landmarks[0]

        zero = hand_landmarks.landmark[mp_hands.HandLandmark(0).value].y
        one = hand_landmarks.landmark[mp_hands.HandLandmark(1).value].y
        five = hand_landmarks.landmark[mp_hands.HandLandmark(5).value].x
        nine = hand_landmarks.landmark[mp_hands.HandLandmark(9).value].x
        thirteen = hand_landmarks.landmark[mp_hands.HandLandmark(13).value].x

        # Điều chỉnh các ngưỡng cho tay người Việt Nam với 10 mức phân loại
        base_heart_x = image_width * (1 - (nine + (five - nine) * 0.35))  
        base_head_x = image_width * (1 - (thirteen + (nine - thirteen) * 0.25))  
        base_life_y = image_height * (one + (zero - one) * 0.25)  

        # Tạo 9 ngưỡng cho mỗi đường để có 10 mức phân loại
        heart_offset = image_width * 0.035  # Giảm offset để có khoảng cách đều hơn
        head_offset = image_width * 0.03
        life_offset = image_height * 0.1
        
        # Tạo 9 ngưỡng với khoảng cách đều
        for i in range(9):
            heart_thres_x[i] = base_heart_x + (i - 4) * heart_offset  # -4 đến +4
            head_thres_x[i] = base_head_x + (i - 4) * head_offset
            life_thres_y[i] = base_life_y + (i - 4) * life_offset

    im = Image.open(path_to_warped_image_mini)
    width = 3
    draw = ImageDraw.Draw(im)
    
    # Kiểm tra nếu lines là None hoặc rỗng
    if lines is None:
        lines = []
    
    # Kiểm tra và xử lý từng đường riêng biệt
    contents = []
    
    # Kiểm tra đường tình cảm
    if len(lines) > 0 and lines[0] is not None and len(lines[0]) > 0:
        heart_line = lines[0]

        # Nội dung mô tả cho 10 mức độ khác nhau
        heart_descriptions = [
            "cực ngắn - bạn vô cùng thận trọng trong tình yêu, cần thời gian dài để tin tưởng ai đó.",
            "rất ngắn - bạn có xu hướng kín đáo, ưa thích tình yêu ổn định và lâu bền.",
            "ngắn - bạn tập trung vào chất lượng hơn số lượng, trân trọng mỗi mối quan hệ.",
            "khá ngắn - bạn cân nhắc kỹ lưỡng trước khi yêu nhưng rất chung thủy.",
            "trung bình - bạn cân bằng hoàn hảo giữa lý trí và cảm xúc trong tình yêu.",
            "khá dài - bạn dễ dàng thể hiện tình cảm và quan tâm đến người khác.",
            "dài - bạn rất cởi mở trong tình yêu, dễ đồng cảm và chia sẻ.",
            "rất dài - bạn có trái tim rộng mở, yêu thương sâu sắc và chân thành.",
            "cực dài - bạn là người lãng mạn cực độ, sống hết mình vì tình yêu.",
            "siêu dài - bạn có tình yêu vô bờ bến, sẵn sàng hy sinh tất cả vì người mình yêu."
        ]
        
        head_descriptions = [
            "cực ngắn - bạn là người hành động, thích quyết định tức thời và theo trực giác.",
            "rất ngắn - bạn ưa thích sự đơn giản, không thích phức tạp hóa vấn đề.",
            "ngắn - bạn tập trung sâu vào một lĩnh vực và trở thành chuyên gia.",
            "khá ngắn - bạn có tư duy thực tế, giải quyết vấn đề một cách hiệu quả.",
            "trung bình - bạn cân bằng giữa tư duy logic và trực giác một cách hoàn hảo.",
            "khá dài - bạn thích tìm hiểu nhiều chủ đề và có khả năng học hỏi nhanh.",
            "dài - bạn có tư duy phân tích tốt, thích nghiên cứu và khám phá.",
            "rất dài - bạn có trí thông minh xuất sắc, khả năng tư duy phức tạp cao.",
            "cực dài - bạn là thiên tài, có khả năng tư duy đa chiều và sáng tạo.",
            "siêu dài - bạn có trí tuệ phi thường, khả năng nhận thức vượt trội."
        ]
        
        life_descriptions = [
            "cực ngắn - bạn cực kỳ độc lập, tự chủ hoàn toàn trong mọi quyết định.",
            "rất ngắn - bạn rất tự lực, ít khi cần sự giúp đỡ từ người khác.",
            "ngắn - bạn thích tự giải quyết vấn đề, có tinh thần tự lập cao.",
            "khá ngắn - bạn độc lập nhưng biết khi nào cần hợp tác với người khác.",
            "trung bình - bạn cân bằng giữa độc lập và hợp tác một cách hoàn hảo.",
            "khá dài - bạn thích làm việc nhóm và tìm kiếm lời khuyên từ người khác.",
            "dài - bạn có năng lượng tốt, thích giao lưu và kết nối với mọi người.",
            "rất dài - bạn có sức sống mạnh mẽ, thích hoạt động xã hội và giúp đỡ người khác.",
            "cực dài - bạn tràn đầy năng lượng, có khả năng lãnh đạo và truyền cảm hứng.",
            "siêu dài - bạn có nguồn năng lượng vô tận, sống hết mình và lan tỏa tích cực."
        ]

    # Kiểm tra đường tình cảm
    if len(lines) > 0 and lines[0] is not None and len(lines[0]) > 0:
        heart_line = lines[0]
        heart_line_points = [tuple(reversed(l[:2])) for l in heart_line]
        heart_line_tip = heart_line_points[0]
        heart_content_1 = 'Đường tình cảm chi phối mọi vấn đề về trái tim, bao gồm tình yêu, tình bạn và cam kết.'
        heart_level = classify_line_length(heart_line_tip[0], heart_thres_x)
        heart_length = calculate_line_length(heart_line_points)
        heart_content_2 = f'Đường tình cảm {heart_descriptions[heart_level]} (chiều dài: {heart_length} pixel)'
        draw.line(heart_line_points, fill="red", width=width)
        contents.extend([heart_content_1, heart_content_2])
    else:
        contents.extend(['Đường tình cảm:', 'Không tìm thấy đường tình cảm trên lòng bàn tay của bạn. Điều này có thể do góc chụp ảnh hoặc chất lượng ảnh không tốt.'])

    # Kiểm tra đường trí tuệ  
    if len(lines) > 1 and lines[1] is not None and len(lines[1]) > 0:
        head_line = lines[1]
        head_line_points = [tuple(reversed(l[:2])) for l in head_line]
        head_line_tip = head_line_points[-1]
        head_content_1 = 'Đường trí tuệ cho biết về sự tò mò trí thức và khả năng tư duy của bạn.'
        head_level = classify_line_length(head_line_tip[0], head_thres_x)
        head_length = calculate_line_length(head_line_points)
        head_content_2 = f'Đường trí tuệ {head_descriptions[head_level]} (chiều dài: {head_length} pixel)'
        draw.line(head_line_points, fill="green", width=width)
        contents.extend([head_content_1, head_content_2])
    else:
        contents.extend(['Đường trí tuệ:', 'Không tìm thấy đường trí tuệ trên lòng bàn tay của bạn. Hãy thử chụp ảnh với ánh sáng tốt hơn và lòng bàn tay thẳng.'])

    # Kiểm tra đường sinh mệnh
    if len(lines) > 2 and lines[2] is not None and len(lines[2]) > 0:
        life_line = lines[2]
        life_line_points = [tuple(reversed(l[:2])) for l in life_line]
        life_line_tip = life_line_points[-1]
        life_content_1 = 'Đường sinh mệnh tiết lộ trải nghiệm, sức sống và nhiệt huyết của bạn. Lưu ý, nó không liên quan đến tuổi thọ!'
        life_level = classify_line_length(life_line_tip[1], life_thres_y)
        life_length = calculate_line_length(life_line_points)
        life_content_2 = f'Đường sinh mệnh {life_descriptions[life_level]} (chiều dài: {life_length} pixel)'
        draw.line(life_line_points, fill="blue", width=width)
        contents.extend([life_content_1, life_content_2])
    else:
        contents.extend(['Đường sinh mệnh:', 'Không tìm thấy đường sinh mệnh trên lòng bàn tay của bạn. Đường này thường rõ nhất, hãy kiểm tra lại ảnh chụp.'])

    return im, contents

def measure_with_structured_data(path_to_warped_image_mini, lines):
    """
    Enhanced measure function that returns both image and structured line data
    """
    from dtos.line_response import LineResponse
    
    # Get the original results
    im, contents = measure(path_to_warped_image_mini, lines)
    
    # Initialize structured line data
    line_responses = []
    
    # Same measurement logic but return structured data
    heart_thres_x = [0] * 9  
    head_thres_x = [0] * 9
    life_thres_y = [0] * 9

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image = cv2.flip(cv2.imread(path_to_warped_image_mini), 1)
        image_height, image_width, _ = image.shape

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # If no hand landmarks detected, return basic data
        if not results.multi_hand_landmarks:
            if lines is None:
                lines = []
            
            # Basic detection without detailed analysis
            if len(lines) > 0 and lines[0] is not None and len(lines[0]) > 0:
                heart_length = calculate_line_length([tuple(reversed(l[:2])) for l in lines[0]])
                line_responses.append(LineResponse("Đường tình cảm", "Phát hiện đường tình cảm nhưng không thể phân tích chi tiết do không nhận dạng được bàn tay.", heart_length))
            
            if len(lines) > 1 and lines[1] is not None and len(lines[1]) > 0:
                head_length = calculate_line_length([tuple(reversed(l[:2])) for l in lines[1]])
                line_responses.append(LineResponse("Đường trí tuệ", "Phát hiện đường trí tuệ nhưng không thể phân tích chi tiết do không nhận dạng được bàn tay.", head_length))
            
            if len(lines) > 2 and lines[2] is not None and len(lines[2]) > 0:
                life_length = calculate_line_length([tuple(reversed(l[:2])) for l in lines[2]])
                line_responses.append(LineResponse("Đường sinh mệnh", "Phát hiện đường sinh mệnh nhưng không thể phân tích chi tiết do không nhận dạng được bàn tay.", life_length))
            
            return im, line_responses

        # Calculate thresholds same as original function
        hand_landmarks = results.multi_hand_landmarks[0]
        zero = hand_landmarks.landmark[mp_hands.HandLandmark(0).value].y
        one = hand_landmarks.landmark[mp_hands.HandLandmark(1).value].y
        five = hand_landmarks.landmark[mp_hands.HandLandmark(5).value].x
        nine = hand_landmarks.landmark[mp_hands.HandLandmark(9).value].x
        thirteen = hand_landmarks.landmark[mp_hands.HandLandmark(13).value].x

        base_heart_x = image_width * (1 - (nine + (five - nine) * 0.35))  
        base_head_x = image_width * (1 - (thirteen + (nine - thirteen) * 0.25))  
        base_life_y = image_height * (one + (zero - one) * 0.25)  

        heart_offset = image_width * 0.035
        head_offset = image_width * 0.03
        life_offset = image_height * 0.1
        
        for i in range(9):
            heart_thres_x[i] = base_heart_x + (i - 4) * heart_offset
            head_thres_x[i] = base_head_x + (i - 4) * head_offset
            life_thres_y[i] = base_life_y + (i - 4) * life_offset

    # Check if lines exists
    if lines is None:
        lines = []
    
    # Description arrays (same as original)
    heart_descriptions = [
        "cực ngắn - bạn vô cùng thận trọng trong tình yêu, cần thời gian dài để tin tưởng ai đó.",
        "rất ngắn - bạn có xu hướng kín đáo, ưa thích tình yêu ổn định và lâu bền.",
        "ngắn - bạn tập trung vào chất lượng hơn số lượng, trân trọng mỗi mối quan hệ.",
        "khá ngắn - bạn cân nhắc kỹ lưỡng trước khi yêu nhưng rất chung thủy.",
        "trung bình - bạn cân bằng hoàn hảo giữa lý trí và cảm xúc trong tình yêu.",
        "khá dài - bạn dễ dàng thể hiện tình cảm và quan tâm đến người khác.",
        "dài - bạn rất cởi mở trong tình yêu, dễ đồng cảm và chia sẻ.",
        "rất dài - bạn có trái tim rộng mở, yêu thương sâu sắc và chân thành.",
        "cực dài - bạn là người lãng mạn cực độ, sống hết mình vì tình yêu.",
        "siêu dài - bạn có tình yêu vô bờ bến, sẵn sàng hy sinh tất cả vì người mình yêu."
    ]
    
    head_descriptions = [
        "cực ngắn - bạn là người hành động, thích quyết định tức thời và theo trực giác.",
        "rất ngắn - bạn ưa thích sự đơn giản, không thích phức tạp hóa vấn đề.",
        "ngắn - bạn tập trung sâu vào một lĩnh vực và trở thành chuyên gia.",
        "khá ngắn - bạn có tư duy thực tế, giải quyết vấn đề một cách hiệu quả.",
        "trung bình - bạn cân bằng giữa tư duy logic và trực giác một cách hoàn hảo.",
        "khá dài - bạn thích tìm hiểu nhiều chủ đề và có khả năng học hỏi nhanh.",
        "dài - bạn có tư duy phân tích tốt, thích nghiên cứu và khám phá.",
        "rất dài - bạn có trí thông minh xuất sắc, khả năng tư duy phức tạp cao.",
        "cực dài - bạn là thiên tài, có khả năng tư duy đa chiều và sáng tạo.",
        "siêu dài - bạn có trí tuệ phi thường, khả năng nhận thức vượt trội."
    ]
    
    life_descriptions = [
        "cực ngắn - bạn cực kỳ độc lập, tự chủ hoàn toàn trong mọi quyết định.",
        "rất ngắn - bạn rất tự lực, ít khi cần sự giúp đỡ từ người khác.",
        "ngắn - bạn thích tự giải quyết vấn đề, có tinh thần tự lập cao.",
        "khá ngắn - bạn độc lập nhưng biết khi nào cần hợp tác với người khác.",
        "trung bình - bạn cân bằng giữa độc lập và hợp tác một cách hoàn hảo.",
        "khá dài - bạn thích làm việc nhóm và tìm kiếm lời khuyên từ người khác.",
        "dài - bạn có năng lượng tốt, thích giao lưu và kết nối với mọi người.",
        "rất dài - bạn có sức sống mạnh mẽ, thích hoạt động xã hội và giúp đỡ người khác.",
        "cực dài - bạn tràn đầy năng lượng, có khả năng lãnh đạo và truyền cảm hứng.",
        "siêu dài - bạn có nguồn năng lượng vô tận, sống hết mình và lan tỏa tích cực."
    ]

    # Check heart line
    if len(lines) > 0 and lines[0] is not None and len(lines[0]) > 0:
        heart_line = lines[0]
        heart_line_points = [tuple(reversed(l[:2])) for l in heart_line]
        heart_line_tip = heart_line_points[0]
        heart_level = classify_line_length(heart_line_tip[0], heart_thres_x)
        heart_length = calculate_line_length(heart_line_points)
        heart_description = f'Đường tình cảm {heart_descriptions[heart_level]} (chiều dài: {heart_length} pixel)'
        line_responses.append(LineResponse("Đường tình cảm", heart_description, heart_length))

    # Check head line
    if len(lines) > 1 and lines[1] is not None and len(lines[1]) > 0:
        head_line = lines[1]
        head_line_points = [tuple(reversed(l[:2])) for l in head_line]
        head_line_tip = head_line_points[-1]
        head_level = classify_line_length(head_line_tip[0], head_thres_x)
        head_length = calculate_line_length(head_line_points)
        head_description = f'Đường trí tuệ {head_descriptions[head_level]} (chiều dài: {head_length} pixel)'
        line_responses.append(LineResponse("Đường trí tuệ", head_description, head_length))

    # Check life line
    if len(lines) > 2 and lines[2] is not None and len(lines[2]) > 0:
        life_line = lines[2]
        life_line_points = [tuple(reversed(l[:2])) for l in life_line]
        life_line_tip = life_line_points[-1]
        life_level = classify_line_length(life_line_tip[1], life_thres_y)
        life_length = calculate_line_length(life_line_points)
        life_description = f'Đường sinh mệnh {life_descriptions[life_level]} (chiều dài: {life_length} pixel)'
        line_responses.append(LineResponse("Đường sinh mệnh", life_description, life_length))

    return im, line_responses